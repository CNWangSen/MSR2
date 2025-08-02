import torch
import numpy

def orthogonal_init(layer, gain=1.0):
	torch.nn.init.orthogonal_(layer.weight, gain=gain)
	torch.nn.init.constant_(layer.bias, 0)
	return layer

class ActorUAVob(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128,using_orthogonal_init=True):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(3,dq_RS)
		self.linear_K_RS=torch.nn.Linear(3,dk_RS)
		self.linear_V_RS=torch.nn.Linear(3,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(3,dq_TG)
		self.linear_K_TG=torch.nn.Linear(3,dk_TG)
		self.linear_V_TG=torch.nn.Linear(3,dv_TG)
	
		self.fc_alpha=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.fc_beta=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	
	def forward(self,s_obs):#batch_size,len_seq,len_features+len_CLS
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		alpha=self.fc_alpha(Hidden)+1.0
		beta=self.fc_beta(Hidden)+1.0
		return alpha,beta
	def get_s_action(self,s_obs,eval=False):
		alpha,beta=self.forward(s_obs)
		#print(mu)
		if(eval):
			alpha1,alpha2=alpha.tolist()[0]
			beta1,beta2=beta.tolist()[0]
			return alpha1/(alpha1+beta1),alpha2/(alpha2+beta2)
		action = torch.distributions.beta.Beta(alpha,beta).rsample().tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_actor_ob_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_actor_ob_"+str(index).zfill(2)+".pth"))
class CriticUAVob(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(3,dq_RS)
		self.linear_K_RS=torch.nn.Linear(3,dk_RS)
		self.linear_V_RS=torch.nn.Linear(3,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(3,dq_TG)
		self.linear_K_TG=torch.nn.Linear(3,dk_TG)
		self.linear_V_TG=torch.nn.Linear(3,dv_TG)

		self.fc_score=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,1),gain=0.01),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	def forward(self,s_obs):
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		score = self.fc_score(Hidden)
		return score
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_critic_ob_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_critic_ob_"+str(index).zfill(2)+".pth"))



class ActorUAVrt(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128,using_orthogonal_init=True):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(3,dq_RS)
		self.linear_K_RS=torch.nn.Linear(3,dk_RS)
		self.linear_V_RS=torch.nn.Linear(3,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(3,dq_TG)
		self.linear_K_TG=torch.nn.Linear(3,dk_TG)
		self.linear_V_TG=torch.nn.Linear(3,dv_TG)
	
		self.fc_alpha=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.fc_beta=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	
	def forward(self,s_obs):#batch_size,len_seq,len_features+len_CLS
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		alpha=self.fc_alpha(Hidden)+1.0
		beta=self.fc_beta(Hidden)+1.0
		return alpha,beta
	def get_s_action(self,s_obs,eval=False):
		alpha,beta=self.forward(s_obs)
		#print(mu)
		if(eval):
			alpha1,alpha2=alpha.tolist()[0]
			beta1,beta2=beta.tolist()[0]
			return alpha1/(alpha1+beta1),alpha2/(alpha2+beta2)
		action = torch.distributions.beta.Beta(alpha,beta).rsample().tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_actor_rt_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_actor_rt_"+str(index).zfill(2)+".pth"))
class CriticUAVrt(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(3,dq_RS)
		self.linear_K_RS=torch.nn.Linear(3,dk_RS)
		self.linear_V_RS=torch.nn.Linear(3,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(3,dq_TG)
		self.linear_K_TG=torch.nn.Linear(3,dk_TG)
		self.linear_V_TG=torch.nn.Linear(3,dv_TG)

		self.fc_score=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,1),gain=0.01),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	def forward(self,s_obs):
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		score = self.fc_score(Hidden)
		return score
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_critic_rt_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_critic_rt_"+str(index).zfill(2)+".pth"))


class ActorUSVrs(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128,using_orthogonal_init=True):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(2,dq_RS)
		self.linear_K_RS=torch.nn.Linear(2,dk_RS)
		self.linear_V_RS=torch.nn.Linear(2,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(2,dq_TG)
		self.linear_K_TG=torch.nn.Linear(2,dk_TG)
		self.linear_V_TG=torch.nn.Linear(2,dv_TG)
	
		self.fc_alpha=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.fc_beta=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act),gain=0.01),
			torch.nn.Softplus(),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	
	def forward(self,s_obs):#batch_size,len_seq,len_features+len_CLS
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		alpha=self.fc_alpha(Hidden)+1.0
		beta=self.fc_beta(Hidden)+1.0
		return alpha,beta
	def get_s_action(self,s_obs,eval=False):
		alpha,beta=self.forward(s_obs)
		#print(mu)
		if(eval):
			alpha1,alpha2=alpha.tolist()[0]
			beta1,beta2=beta.tolist()[0]
			return alpha1/(alpha1+beta1),alpha2/(alpha2+beta2)
		action = torch.distributions.beta.Beta(alpha,beta).rsample().tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_actor_rs_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_actor_rs_"+str(index).zfill(2)+".pth"))
class CriticUSVrs(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		dq_RS=16
		dk_RS=16
		dv_RS=32
		dq_TG=16
		dk_TG=16
		dv_TG=32

		self.linear_Q_RS=torch.nn.Linear(2,dq_RS)
		self.linear_K_RS=torch.nn.Linear(2,dk_RS)
		self.linear_V_RS=torch.nn.Linear(2,dv_RS)

		self.linear_Q_TG=torch.nn.Linear(2,dq_TG)
		self.linear_K_TG=torch.nn.Linear(2,dk_TG)
		self.linear_V_TG=torch.nn.Linear(2,dv_TG)

		self.fc_score=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dv_TG+dv_RS,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,1),gain=0.01),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	def forward(self,s_obs):
		#Num_RS = (int)(s_obs[0,0,2].item())
		#Num_TG = (int)(s_obs[0,1,2].item())
		#obs_RS = s_obs[:,0:Num_RS,0:2]#batsh_size,len_seq,len_features
		#obs_TG = s_obs[:,Num_RS:Num_RS+Num_TG,0:2]#batsh_size,len_seq,len_features

		Q_RS=self.linear_Q_RS(s_obs)#batch_size,len_seq,dq
		K_RS=self.linear_K_RS(s_obs)#batch_size,len_seq,dk
		V_RS=self.linear_V_RS(s_obs)#batch_size,len_seq,dv
		QK_RS=torch.bmm(Q_RS,K_RS.transpose(1,2))/(K_RS.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_RS=torch.bmm(torch.nn.functional.softmax(QK_RS,dim=2),V_RS)#batch_size,len_seq,dv	

		Q_TG=self.linear_Q_TG(s_obs)#batch_size,len_seq,dq
		K_TG=self.linear_K_TG(s_obs)#batch_size,len_seq,dk
		V_TG=self.linear_V_TG(s_obs)#batch_size,len_seq,dv
		QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
		QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv	

		Hidden_TG = torch.mean(QKV_TG,dim=1)
		Hidden_RS = torch.mean(QKV_RS,dim=1)
		Hidden=torch.cat((Hidden_RS,Hidden_TG),dim=1)

		score = self.fc_score(Hidden)
		return score
	def save_checkpoint(self,checkpoint_file_prefix,index):
		torch.save(self.state_dict(),checkpoint_file_prefix+"_critic_rs_"+str(index).zfill(2)+".pth")
	def load_checkpoint(self,checkpoint_file_prefix,index):
		self.load_state_dict(torch.load(checkpoint_file_prefix+"_critic_rs_"+str(index).zfill(2)+".pth"))
