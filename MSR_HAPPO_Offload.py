from env.msr import MyWrapper,MSR
from env.const import *
from env.trick import *
from model.MSR_agent_MAPPO_Transformer import ActorUAVob,ActorUAVrt,ActorUSVrs,CriticCentral
from model.MSR_agent_MAPPO_Offload import ActorUAVob_offload,ActorUAVrt_offload,ActorUSVrs_offload,CriticCentral_offload
import torch
import numpy
import os
import random
import time
#fetch
IS_TRAINING=0
CONTINUE=0

class MSR_HAPPO_Offload:
	def __init__(self,NUM_EPOCH=300000,RUN_ID=602,device ="cuda:1",collecting_paper_data=False,
			  parameter_sharing=False,using_advantage_norm=True,using_state_norm=False,
			  using_entropy_loss=True,using_learning_rate_decay=True,using_gradient_clip=True,
			  rt_num=2,ob_num=3,rs_num=1,tg_num=80,
			  stochastic_action=True,map_scaling=1,current_amp=1):
		self.NUM_EPOCH=NUM_EPOCH
		self.RUN_ID=RUN_ID
		self.device=device

		self.alg_name="HAPPO_Offload"
		self.parameter_sharing=parameter_sharing
		self.using_advantage_norm=using_advantage_norm
		self.using_state_norm=using_state_norm
		self.using_entropy_loss=using_entropy_loss
		self.using_learning_rate_decay=using_learning_rate_decay
		self.using_gradient_clip=using_gradient_clip

		self.stochastic_action=stochastic_action
		self.map_scaling=map_scaling
		self.current_amp=current_amp

		self.best_score=-1000000

		self.wp=MyWrapper(use_group_reward=True,alg_name=self.alg_name,
					rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num,
					map_scaling=map_scaling,current_amp=current_amp)
		self.wp.reset()
		self.wp.env.is_training_offload=True

		self.actors=[]
		self.critics=[]
		self.actors_offload=[]
		self.critics_offload=[]
		
		if(self.parameter_sharing):
			actor_cls=[ActorUAVob().to(self.device),ActorUAVrt().to(self.device),ActorUSVrs().to(self.device)]
			critic_cls=[CriticCentral().to(self.device)]
			for p in self.wp.env.players:
				self.actors.append(actor_cls[p.cls])
				self.critics.append(critic_cls[0])
			actor_cls_offload=[ActorUAVob_offload().to(self.device),ActorUAVrt_offload().to(self.device),ActorUSVrs_offload().to(self.device)]
			critic_cls_offload=[CriticCentral_offload().to(self.device)]
			for p in self.wp.env.players:
				self.actors_offload.append(actor_cls_offload[p.cls])
				self.critics_offload.append(critic_cls_offload[0])
		else:
			actor_cls=[ActorUAVob,ActorUAVrt,ActorUSVrs]
			critic_cls=[CriticCentral]
			for p in self.wp.env.players:
				self.actors.append(actor_cls[p.cls]().to(self.device))
			self.critics=[critic_cls[0]().to(self.device)]
			actor_cls_offload=[ActorUAVob_offload,ActorUAVrt_offload,ActorUSVrs_offload]
			critic_cls_offload=[CriticCentral_offload]
			for p in self.wp.env.players:
				self.actors_offload.append(actor_cls_offload[p.cls]().to(self.device))
			self.critics_offload=[critic_cls_offload[0]().to(self.device)]
		if(self.using_state_norm):
			self.state_norm_buffer_by_agent=[]
			for p in self.wp.env.players:
				self.state_norm_buffer_by_agent.append(RunningMeanStd(dim_obs=p.dim_s_obs_offload))

		self.collecting_paper_data=collecting_paper_data
		
		self.paper_data_J_rewards_train=[[] for i in range(len(self.wp.env.players))]
		self.paper_data_save_cnt_train=[]
		self.paper_data_po_update_cnt_train=[]
		self.paper_data_ob_update_cnt_train=[]
		self.paper_data_tg_pos_corrected_train=[]
		self.paper_data_J_actor_loss_train=[[] for i in range(len(self.wp.env.players))]
		self.paper_data_J_critic_loss_train=[[] for i in range(len(self.wp.env.players))]
		
		self.paper_data_J_rewards_eval=[[] for i in range(len(self.wp.env.players))]
		self.paper_data_T_10_eval=[]
		self.paper_data_T_20_eval=[]
		self.paper_data_T_30_eval=[]
		self.paper_data_T_40_eval=[]
		self.paper_data_T_50_eval=[]
		self.paper_data_T_60_eval=[]
		self.paper_data_T_70_eval=[]
		self.paper_data_T_80_eval=[]
		self.paper_data_T_90_eval=[]
		self.paper_data_T_100_eval=[]
		self.paper_data_Final_Percent_eval=[]
		self.paper_data_po_update_cnt_eval=[]
		self.paper_data_ob_update_cnt_eval=[]
		self.paper_data_tg_pos_corrected_eval=[]

		self.do_not_save=False
		self.epoch=0
	def save(self):
		if(self.do_not_save):
			return
		print("-HAPPO-SAVE",time.ctime())
		try:
			os.mkdir('pth/'+self.alg_name+"/")
		except:
			pass
		for i in range(len(self.actors_offload)):
			index = i
			self.actors_offload[i].save_checkpoint('pth/'+self.alg_name+'/'+str(self.RUN_ID)+"_"+self.alg_name,index)
		for i in range(len(self.critics_offload)):
			index = i
			self.critics_offload[i].save_checkpoint('pth/'+self.alg_name+'/'+str(self.RUN_ID)+"_"+self.alg_name,index)
		print("SAVE!")

	def load(self):
		rt_cnt=0
		ob_cnt=0
		rs_cnt=0
		
		for i in range(len(self.actors_offload)):
			#print("loading"+str(i))
			player_cls=self.wp.env.players[i].cls
			if(player_cls==ENUM_CLS_RT):
				index=rt_cnt%2+0
				rt_cnt+=1
			if(player_cls==ENUM_CLS_OB):
				index=ob_cnt%3+2
				ob_cnt+=1
			if(player_cls==ENUM_CLS_RS):
				index=rs_cnt%1+5
				rs_cnt+=1
			self.actors_offload[i].load_checkpoint('pth/'+self.alg_name+'/'+str(self.RUN_ID)+"_"+self.alg_name,index,self.device)
		self.critics_offload[0].load_checkpoint('pth/'+self.alg_name+'/'+str(self.RUN_ID)+"_"+self.alg_name,0,self.device)

		rt_cnt=0
		ob_cnt=0
		rs_cnt=0
		
		for i in range(len(self.actors)):
			#print("loading"+str(i))
			player_cls=self.wp.env.players[i].cls
			if(player_cls==ENUM_CLS_RT):
				index=rt_cnt%2+0
				rt_cnt+=1
			if(player_cls==ENUM_CLS_OB):
				index=ob_cnt%3+2
				ob_cnt+=1
			if(player_cls==ENUM_CLS_RS):
				index=rs_cnt%1+5
				rs_cnt+=1
			self.actors[i].load_checkpoint('pth/HAPPO_Transformer/'+str(self.RUN_ID)+"_HAPPO_Transformer",index,self.device)
		self.critics[0].load_checkpoint('pth/HAPPO_Transformer/'+str(self.RUN_ID)+"_HAPPO_Transformer",0,self.device)	
		print("LOAD!")

	def state_norm(self,J_obs,eval=False):
		if(self.using_state_norm):
			for i in range(len(self.state_norm_buffer_by_agent)):
				if(eval==False):
					self.state_norm_buffer_by_agent[i].update(J_obs[i])
				
				for j in range(len(J_obs[i])):
					J_obs[i][j]=(J_obs[i][j]-self.state_norm_buffer_by_agent[i].mean[j])/(self.state_norm_buffer_by_agent[i].std[j]+1e-8)
			
		return J_obs

	def get_J_action(self,J_obs,eval=False):
		J_action=[]
		for i in range(len(J_obs)):
			tensor_obs=torch.FloatTensor(J_obs[i]).to(self.device)
			len_feature=self.wp.env.players[i].dim_feature
			len_seq=(int)(tensor_obs.shape[0]/len_feature)
			J_action.append(
				self.actors_offload[i].get_s_action(tensor_obs.reshape(1,len_seq,len_feature),eval=(not self.stochastic_action))
			)
		return J_action
	
	def get_traj(self,show=False,eval=False,reset_pos=True):
		if(show):
			try:
				os.mkdir('img/'+self.alg_name+'/')
			except:
				pass
			for f in os.listdir('img/'+self.alg_name+'/'):
				os.remove('img/'+self.alg_name+'/'+f)
		if(not eval):
			#[len_agent,tensor(len_traj,dim_obs)]
			J_obses_tensor=[torch.zeros(self.wp.env.max_traj_length,self.wp.env.players[i].dim_s_obs_offload).to(self.device) for i in range(self.wp.env.N)]
			J_states_tensor=[torch.zeros(self.wp.env.max_traj_length,(len(self.wp.env.players)+len(self.wp.env.npcs))*3).to(self.device)]
			J_actions_tensor=[torch.zeros(self.wp.env.max_traj_length,self.wp.env.players[i].dim_s_act_offload).to(self.device) for i in range(self.wp.env.N)]
			J_rewards_tensor=[torch.zeros(self.wp.env.max_traj_length,1).to(self.device) for i in range(self.wp.env.N)]
			J_next_obses_tensor=[torch.zeros(self.wp.env.max_traj_length,self.wp.env.players[i].dim_s_obs_offload).to(self.device) for i in range(self.wp.env.N)]
			J_next_states_tensor=[torch.zeros(self.wp.env.max_traj_length,(len(self.wp.env.players)+len(self.wp.env.npcs))*3).to(self.device)]
			J_dones_tensor=[torch.zeros(self.wp.env.max_traj_length,1).to(self.device) for i in range(self.wp.env.N)]

		J_obs=self.wp.reset(reset_pos=reset_pos)
		if(eval==False):
			J_obs=self.state_norm(J_obs,eval=eval)
		J_obs_offload=self.wp.reset_offload(reset_pos=reset_pos)
		if(eval==False):
			J_obs_offload=self.state_norm(J_obs_offload,eval=eval)

		reward_ep=0.0
		J_traj_reward_ep_eval=[0.0 for actor_i in range(len(self.wp.env.players))]
		J_traj_reward_ep_train=[0.0 for actor_i in range(len(self.wp.env.players))]
		for traj_i in range(self.wp.env.max_traj_length):
			J_state =[]
			J_next_state = [] 
			if(not eval):
				J_state=self.get_states()
			J_action=self.get_J_action(J_obs,eval=eval)
			J_next_obs,J_reward,J_done=self.wp.step(J_action,eval=eval)
			J_action_offload = self.get_J_action_offload(J_obs,eval=eval)
			J_next_obs_offload,J_reward_offload,J_done_offload=self.wp.step_offload(J_action_offload,eval=eval)
			if(eval==False):
				J_next_obs=self.state_norm(J_next_obs,eval=eval)
				J_next_obs_offload=self.state_norm(J_next_obs_offload,eval=eval)
			#J_next_action=self.get_J_action(J_next_obs)

			if(not eval):
				for actor_i in range(len(self.wp.env.players)):
					J_obses_tensor[actor_i][traj_i]=torch.FloatTensor(J_obs_offload[actor_i])
					J_actions_tensor[actor_i][traj_i]=torch.FloatTensor(J_action_offload[actor_i])
					J_rewards_tensor[actor_i][traj_i]=J_reward_offload[actor_i]
					J_next_obses_tensor[actor_i][traj_i]=torch.FloatTensor(J_next_obs_offload[actor_i])
					J_dones_tensor[actor_i][traj_i]=1 if J_done_offload[actor_i] else 0
				J_next_state=self.get_states()
				J_states_tensor[0][traj_i]=torch.FloatTensor(J_state)
				J_next_states_tensor[0][traj_i]=torch.FloatTensor(J_next_state)
			if(self.collecting_paper_data):
				if(eval):
					for actor_i in range(len(self.wp.env.players)):
						J_traj_reward_ep_eval[actor_i]+=J_reward[actor_i]
				else:
					for actor_i in range(len(self.wp.env.players)):
						J_traj_reward_ep_train[actor_i]+=J_reward[actor_i]
			J_obs=J_next_obs
			J_obs_offload=J_next_obs_offload
			reward_ep+=J_reward[0]
			if(show):
				self.wp.show()
			if(J_done[0]):
				break
		if(eval):
			if(self.collecting_paper_data):
				for actor_i in range(len(self.wp.env.players)):
					self.paper_data_J_rewards_eval[actor_i].append(J_traj_reward_ep_eval[actor_i])
			return reward_ep
		else:
			if(self.collecting_paper_data):
				for actor_i in range(len(self.wp.env.players)):
					self.paper_data_J_rewards_train[actor_i].append(J_traj_reward_ep_train[actor_i])
			return traj_i,J_states_tensor,J_next_states_tensor,J_obses_tensor,J_actions_tensor,J_rewards_tensor,J_next_obses_tensor,J_dones_tensor

	def get_states(self):
		J_state=[]
		cnt_rt=0
		for p in self.wp.env.players:
			if(p.cls==ENUM_CLS_RT):			
				J_state.append(p.x)
				J_state.append(p.y)
				J_state.append(1)
				cnt_rt+=1
		
		cnt_ob=0
		for p in self.wp.env.players:
			if(p.cls==ENUM_CLS_OB):			
				J_state.append(p.x)
				J_state.append(p.y)
				J_state.append(1)
				cnt_ob+=1
		
		cnt_rs=0
		for p in self.wp.env.players:
			if(p.cls==ENUM_CLS_RS):			
				J_state.append(p.x)
				J_state.append(p.y)
				J_state.append(1)
				cnt_rs+=1
		
		cnt_tg=0
		for n in self.wp.env.npcs:
			if(n.is_saved==0):
				J_state.append(n.x)
				J_state.append(n.y)
				J_state.append(0)
			else:
				J_state.append(0)
				J_state.append(0)
				J_state.append(0)
			cnt_tg+=1
		
		J_state[2]=cnt_rt
		J_state[5]=cnt_ob
		J_state[8]=cnt_rs
		J_state[11]=cnt_tg
		return J_state
	def get_advantages(self,deltas):
		advantages=[]
		deltas=deltas.squeeze(dim=1).tolist()
		s=0.0
		for d in deltas[::-1]:
			s=0.98*0.95*s+d
			advantages.append(s)
		advantages.reverse()#list, len = batch size
		if(self.using_advantage_norm):
			mu = numpy.mean(advantages)
			std = numpy.std(advantages)+1e-8
			advantages -= mu
			advantages /= std
		advantages=torch.FloatTensor(advantages).reshape(-1,1).to(self.device)
		return advantages

	def sample_batch(self,traj_len,batch_size,states_traj,next_states_traj,J_obses_traj,J_actions_traj,J_rewards_traj,J_next_obses_traj,J_dones_traj):

		states_sampled=[]
		next_states_sampled=[]
		J_obses_sampled=[]
		J_actions_sampled=[]
		J_rewards_sampled=[]
		J_next_obses_sampled=[]
		J_dones_sampled=[]
		for actor_i in range(len(self.wp.env.players)):
			len_feature=self.wp.env.players[actor_i].dim_feature
			len_seq=(int)(J_obses_traj[actor_i].shape[1]/len_feature)
			
			J_obses_sampled.append(J_obses_traj[actor_i][0:traj_len,:].reshape(traj_len,len_seq,len_feature))
			J_actions_sampled.append(J_actions_traj[actor_i][0:traj_len,:])
			J_rewards_sampled.append(J_rewards_traj[actor_i][0:traj_len,:])
			J_next_obses_sampled.append(J_next_obses_traj[actor_i][0:traj_len,:].reshape(traj_len,len_seq,len_feature))
			J_dones_sampled.append(J_dones_traj[actor_i][0:traj_len,:])
		
		len_feature=3
		len_seq=(int)(states_traj[0].shape[1]/len_feature)
		states_sampled.append(states_traj[0][0:traj_len,:].reshape(traj_len,len_seq,len_feature))

		len_feature=3
		len_seq=(int)(next_states_traj[0].shape[1]/len_feature)
		next_states_sampled.append(next_states_traj[0][0:traj_len,:].reshape(traj_len,len_seq,len_feature))
		actual_batch_size=traj_len
		return actual_batch_size,states_sampled,next_states_sampled,J_obses_sampled,J_actions_sampled,J_rewards_sampled,J_next_obses_sampled,J_dones_sampled

	def train(self,save_train=True,train_num=0):
		if(train_num==0):
			train_num=self.NUM_EPOCH
		train_start=self.epoch
		train_end=self.epoch+train_num+1
		loss_fn=torch.nn.MSELoss()
		for ep in range(train_start,train_end):
			self.epoch=ep
			traj_len,states_traj,next_states_traj,J_obses_traj,J_actions_traj,J_rewards_traj,J_next_obses_traj,J_dones_traj=self.get_traj(show=False,eval=False)

			actual_batch_size,states,next_states,J_obses,J_actions,J_rewards,J_next_obses,J_dones=self.sample_batch(traj_len,256,states_traj,next_states_traj,J_obses_traj,J_actions_traj,J_rewards_traj,J_next_obses_traj,J_dones_traj)
			if(self.using_learning_rate_decay):
				for p in self.critics_offload[0].optimizer.param_groups:
					p['lr'] = 1e-3*(1-self.epoch/(self.NUM_EPOCH+1))

			values=self.critics_offload[0](states[0]).detach()
			targets=self.critics_offload[0](next_states[0]).detach()
			targets*=0.98
			targets*=(1-J_dones[0])
			targets+=J_rewards[0]
			advantages=self.get_advantages(targets-values)
	
			for i in range(len(self.actors_offload)):

				if(self.using_learning_rate_decay):
					for p in self.actors_offload[i].optimizer.param_groups:
						p['lr'] = 1e-3*(1-self.epoch/(self.NUM_EPOCH+1))



				alpha,beta=self.actors_offload[i](J_obses[i])
				old_probs=torch.distributions.beta.Beta(alpha,beta)		
				old_probs=old_probs.log_prob(J_actions[i])
				old_probs=old_probs.exp().detach()

				#for _ in range(1):
				alpha,beta=self.actors_offload[i](J_obses[i])
				new_dis=torch.distributions.beta.Beta(alpha,beta)
				new_probs=new_dis.log_prob(J_actions[i])
				new_probs=new_probs.exp()

				ratios=new_probs/old_probs
				surr1s=ratios*advantages
				surr2s=torch.clamp(ratios,0.8,1.2)*advantages

				actor_loss=-torch.min(surr1s,surr2s)
				if(self.using_entropy_loss):
					entropy = new_dis.entropy().sum(1, keepdim=True)
					actor_loss -= 0.01*entropy
				actor_loss=actor_loss.mean()

				self.actors_offload[i].optimizer.zero_grad()
				actor_loss.backward()
				if(self.using_gradient_clip):
					torch.nn.utils.clip_grad_norm_(self.actors_offload[i].parameters(), 0.5) 
				self.actors_offload[i].optimizer.step()



				alpha,beta=self.actors_offload[i](J_obses[i])
				new_dis=torch.distributions.beta.Beta(alpha,beta)
				new_probs=new_dis.log_prob(J_actions[i])
				new_probs=new_probs.exp()

				ratios=new_probs/old_probs
				advantages=ratios.detach()*advantages
				if(self.collecting_paper_data):
					self.paper_data_J_actor_loss_train[i].append(actor_loss.detach().cpu().item())
					
 
			values=self.critics_offload[0](states[0])
			critic_loss=loss_fn(values,targets)
			if(self.collecting_paper_data):
				for i in range(len(self.actors_offload)):
					self.paper_data_J_critic_loss_train[i].append(critic_loss.detach().cpu().item())
	 
			self.critics_offload[0].optimizer.zero_grad()
			critic_loss.backward()
			if(self.using_gradient_clip):
				torch.nn.utils.clip_grad_norm_(self.critics_offload[0].parameters(), 0.5) 
			self.critics_offload[0].optimizer.step()

	 
			if(self.collecting_paper_data):
				self.paper_data_J_rewards_train.append(J_rewards)
				self.paper_data_save_cnt_train.append(self.wp.env.save_cnt)
				self.paper_data_po_update_cnt_train.append(self.wp.env.po_update_cnt)
				self.paper_data_ob_update_cnt_train.append(self.wp.env.ob_update_cnt)
				self.paper_data_tg_pos_corrected_train.append(self.wp.env.tg_pos_corrected)

				self.get_traj(show=False,eval=True)
				self.paper_data_T_10_eval.append(self.wp.env.T_10)
				self.paper_data_T_20_eval.append(self.wp.env.T_20)
				self.paper_data_T_30_eval.append(self.wp.env.T_30)
				self.paper_data_T_40_eval.append(self.wp.env.T_40)
				self.paper_data_T_50_eval.append(self.wp.env.T_50)
				self.paper_data_T_60_eval.append(self.wp.env.T_60)
				self.paper_data_T_70_eval.append(self.wp.env.T_70)
				self.paper_data_T_80_eval.append(self.wp.env.T_80)
				self.paper_data_T_90_eval.append(self.wp.env.T_90)
				self.paper_data_T_100_eval.append(self.wp.env.T_100)
				self.paper_data_Final_Percent_eval.append(self.wp.env.Final_Percent)
				self.paper_data_po_update_cnt_eval.append(self.wp.env.po_update_cnt)
				self.paper_data_ob_update_cnt_eval.append(self.wp.env.ob_update_cnt)
				self.paper_data_tg_pos_corrected_eval.append(self.wp.env.tg_pos_corrected)
				if(self.epoch%100==0):
					self.save()
			elif self.epoch%100==0:
				with torch.no_grad():
					CR=0.0
					Final_Percent=0
					T_100=0
					for _ in range(10):
						CR+=self.get_traj(show=False,eval=True)
						Final_Percent+=self.wp.env.Final_Percent
						if self.wp.env.T_100!=0:
							T_100+=self.wp.env.T_100
						else:
							T_100+=10000 
					print(self.epoch,CR/10,Final_Percent/10,T_100/10)
					if(-T_100/10>=self.best_score):
						self.save()
						self.best_score=-T_100/10

if __name__ == '__main__':
	t=MSR_HAPPO()
	if IS_TRAINING:
		if CONTINUE:
			t.load()
		t.train()
	else:
		t.load()
		print(t.get_traj(show=True,eval=True))
