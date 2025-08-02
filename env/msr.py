from .npc import npc_target
from .player import player_UAV_ob,player_UAV_rt,player_USV_rs
from .tool import t_dis
import random
import matplotlib as mpl
from matplotlib import pyplot as plt
from .const import *
import numpy
import pandas as pd

class MyWrapper:
	def __init__(self,use_group_reward=False,alg_name="IPPO",rt_num=2,ob_num=3,rs_num=1,tg_num=80,map_scaling=1,current_amp=1):
		self.env=MSR(use_group_reward=use_group_reward,alg_name=alg_name,rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num,map_scaling=map_scaling,current_amp=current_amp)
	def reset(self,reset_pos=True):
		J_obs = self.env.reset(reset_pos=reset_pos)
		return J_obs
	def reset_offload(self,reset_pos=True):
		J_obs_offload = self.env.reset_offload(reset_pos=reset_pos)
		return J_obs_offload
	def step(self,J_action,eval=False):
		J_next_obs,J_reward,J_done=self.env.step(J_action,eval=eval)
		return J_next_obs,J_reward,J_done
	def step_offload(self,J_action_offload,eval=False):
		J_next_obs_offload,J_reward_offload,J_done_offload=self.env.step_offload(J_action_offload,eval=eval)
		return J_next_obs_offload,J_reward_offload,J_done_offload		
	def show(self):
		self.env.render()

class MSR():
	def GetClone(self):
		CloneEnv=MSR()
		return CloneEnv
	def __init__(self,use_group_reward=True,alg_name="IPPO",rt_num=2,ob_num=3,rs_num=1,tg_num=80,map_scaling=1,current_amp=1):
		self.max_traj_length=2160
		self.alg_name=alg_name
		self.players=[]
		self.is_po=False
		for i in range(rt_num):
			self.players.append(player_UAV_rt(rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num))
			self.is_po=True
		for i in range(ob_num):
			self.players.append(player_UAV_ob(rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num))
		for i in range(rs_num):
			self.players.append(player_USV_rs(rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num))
		self.npcs=[npc_target() for i in range(tg_num)]

		self.players_ob=[]
		self.players_rt=[]
		self.players_rs=[]
		self.dim_J_obs=[]
		self.dim_J_act=[]
		self.dim_J_reward=[]
		self.dim_state=0
		self.map_scaling=map_scaling
		self.current_amp=current_amp
		self.Xmin=self.map_scaling*ENV_BOUND_X_MIN
		self.Xmax=self.map_scaling*ENV_BOUND_X_MAX
		self.Ymin=self.map_scaling*ENV_BOUND_Y_MIN
		self.Ymax=self.map_scaling*ENV_BOUND_Y_MAX

		
		self.use_group_reward=use_group_reward

		cnt=0
		for p in self.players:
			p.env=self
			self.dim_J_obs.append(p.dim_s_obs)
			self.dim_J_act.append(p.dim_s_act)
			self.dim_J_reward.append(1)
			self.dim_state+=(p.dim_s_obs+p.dim_s_act)

			if(p.cls==ENUM_CLS_OB):
				for n in self.npcs:
					p.in_my_focus.append(n)
				self.players_ob.append(p)
			elif(p.cls==ENUM_CLS_RT):
				for p2 in self.players:
					p.in_my_focus.append(p2)
				self.players_rt.append(p)
			elif(p.cls==ENUM_CLS_RS):
				for n in self.npcs:
					p.in_my_focus.append(n)
				self.players_rs.append(p)
			else:
				print("CLS NOT FOUND")#可以log出来 undefined情况
				pass
			p.guid=cnt
			cnt+=1
			
		for n in self.npcs:
			n.env=self
			n.guid=cnt
			cnt+=1
		self.N=len(self.players)
		self.env_steps=0
		self.current_dir=0
		self.links=[]

		self.T_10=0
		self.T_20=0
		self.T_30=0
		self.T_40=0
		self.T_50=0
		self.T_60=0
		self.T_70=0
		self.T_80=0
		self.T_90=0
		self.T_100=0
		self.Final_Percent=0
		self.save_cnt=0
		self.po_update_cnt=0
		self.ob_update_cnt=0
		self.tg_pos_corrected=0
		self.is_training_offload=False
	def reset(self,reset_pos=True):
		startx,starty=self.getRandomCircleXY(0.7,1)
		for p in self.players:
			p.reset(reset_pos=reset_pos)
			if(reset_pos):
				p.x,p.y=startx,starty
		for n in self.npcs:
			n.reset(reset_pos=reset_pos)
		self.env_steps=0
		self.current_dir=random.random()*360/UNIT_DEGREE
  		

		self.T_10=0
		self.T_20=0
		self.T_30=0
		self.T_40=0
		self.T_50=0
		self.T_60=0
		self.T_70=0
		self.T_80=0
		self.T_90=0
		self.T_100=0
		self.Final_Percent=0
		self.save_cnt=0
		self.po_update_cnt=0
		self.ob_update_cnt=0
		self.tg_pos_corrected=0
		return self.Get_J_obs()
	def reset_offload(self,reset_pos=True):
		startx,starty=self.getRandomCircleXY(0.7,1)
		for p in self.players:
			p.reset(reset_pos=reset_pos)
			if(reset_pos):
				p.x,p.y=startx,starty
		for n in self.npcs:
			n.reset(reset_pos=reset_pos)
		self.env_steps=0
		self.current_dir=random.random()*360/UNIT_DEGREE
  		

		self.T_10=0
		self.T_20=0
		self.T_30=0
		self.T_40=0
		self.T_50=0
		self.T_60=0
		self.T_70=0
		self.T_80=0
		self.T_90=0
		self.T_100=0
		self.Final_Percent=0
		self.save_cnt=0
		self.po_update_cnt=0
		self.ob_update_cnt=0
		self.tg_pos_corrected=0
		return self.Get_J_obs_offload()
	@property
	def t(self):
		return self.env_steps*UNIT_TIME

	def Get_J_obs(self,eval=False):
		J_obs=[]
		for p in self.players:
			J_obs.append(p.get_s_obs(eval=eval))
		return J_obs
	def Get_J_obs_offload(self,eval=False):
		J_obs_offload=[]
		for p in self.players:
			J_obs_offload.append(p.get_s_obs_offload(eval=eval))
		return J_obs_offload


	def Get_J_reward(self,eval=False):
		J_reward=[]
		if(self.use_group_reward):
			Group_Reward=0
			for p in self.players:
				Group_Reward+=p.penalty_this_step
				p.penalty_this_step=0
			for p in self.players:
				J_reward.append(Group_Reward)
		else:
			for p in self.players:
				Single_Reward=p.penalty_this_step
				J_reward.append(Single_Reward)
				p.penalty_this_step=0
		return J_reward
	def Get_J_reward_offload(self,eval=False):
		J_reward_offload=[]
		if(self.use_group_reward):
			Group_Reward=0
			for p in self.players:
				Group_Reward+=p.penalty_this_step
				p.penalty_this_step=0
			for p in self.players:
				J_reward_offload.append(Group_Reward)
		else:
			for p in self.players:
				Single_Reward=p.penalty_this_step
				J_reward_offload.append(Single_Reward)
				p.penalty_this_step=0
		return J_reward_offload
	def Get_J_done(self,eval=False):
		J_done=[]
		done=True
		for n in self.npcs:
			if(n.is_saved==0):
				done=False
		if(self.env_steps>=self.max_traj_length):
			done=True
		for p in self.players:
			J_done.append(done)
		return J_done
	def Get_J_done_offload(self,eval=False):
		J_done_offload=[]
		done=True
		for n in self.npcs:
			if(n.is_saved==0):
				done=False
		if(self.env_steps>=self.max_traj_length):
			done=True
		for p in self.players:
			J_done_offload.append(done)
		return J_done_offload
	def step_J_action(self,J_action,eval=False):
		for i in range(len(self.players)):
			self.players[i].step_s_action(J_action[i],eval=eval)
		for n in self.npcs:
			n.step_s_action([],eval=eval)
	def step_J_action_offload(self,J_action_offload,eval=False):
		for i in range(len(self.players)):
			self.players[i].step_s_action_offload(J_action_offload[i],eval=eval)
		for n in self.npcs:
			n.step_s_action_offload([],eval=eval)
	
	def getRandomCircleXY(self,r1,r2):
		r=random.uniform(r1*self.map_scaling,r2*self.map_scaling)
		t=random.uniform(0,2*math.pi)
		return r*math.cos(t),r*math.sin(t)

	def step(self,J_action,eval=False):
		self.links=[]
		self.step_J_action(J_action,eval=eval)

		if(self.is_po):
			self.update_po_info_bylinks()
		else:
			self.update_po_info_bygod()
	
		J_next_obs = self.Get_J_obs(eval=eval)
		J_reward = self.Get_J_reward(eval=eval)
		J_done = self.Get_J_done(eval=eval)
  
		Rescued=0
		for n in self.npcs:
			if(n.is_saved==1):
				Rescued+=1
		Rescued_Percent=Rescued/len(self.npcs)

		self.env_steps+=1
		if(self.T_10==0 and Rescued_Percent>=0.10):
			self.T_10=self.env_steps
		if(self.T_20==0 and Rescued_Percent>=0.20):
			self.T_20=self.env_steps
		if(self.T_30==0 and Rescued_Percent>=0.30):
			self.T_30=self.env_steps
		if(self.T_40==0 and Rescued_Percent>=0.40):
			self.T_40=self.env_steps
		if(self.T_50==0 and Rescued_Percent>=0.50):
			self.T_50=self.env_steps
		if(self.T_60==0 and Rescued_Percent>=0.60):
			self.T_60=self.env_steps
		if(self.T_70==0 and Rescued_Percent>=0.70):
			self.T_70=self.env_steps
		if(self.T_80==0 and Rescued_Percent>=0.80):
			self.T_80=self.env_steps
		if(self.T_90==0 and Rescued_Percent>=0.90):
			self.T_90=self.env_steps
		if(self.T_100==0 and Rescued_Percent>=1.00):
			self.T_100=self.env_steps
		self.Final_Percent=Rescued_Percent
		return J_next_obs,J_reward,J_done
	def step_offload(self,J_action_offload,eval=False):
		self.links=[]
		self.step_J_action_offload(J_action_offload,eval=eval)

		if(self.is_po):
			self.update_po_info_bylinks()
		else:
			self.update_po_info_bygod()
	
		J_next_obs = self.Get_J_obs_offload(eval=eval)
		J_reward = self.Get_J_reward_offload(eval=eval)
		J_done = self.Get_J_done_offload(eval=eval)
  
		self.env_steps+=1
		return J_next_obs,J_reward,J_done
	def render(self):

		#plt.rcParams['font.family'] = 'Times New Roman'
		#plt.rcParams['font.size'] = 7.5  # 对应6号字
		#mpl.rcParams['svg.fonttype'] = 'none'
		#mpl.rcParams['svg.hashsalt'] = 'hello'
		#DPI=72#300
		#COLORS=[
		#"#c72228","#f98f34","#0c4e9b",
		#"#f5867f","#ffbc80","#6b98c4"
		#]

		#PHYWIDTH=2.047/5.2*5.2
		#PHYHEIGHT=1.532/5.2*5.2
		#LEG_WIDTH=1
		#LINE_WIDTH=1
		#MARKER_SIZE=2.5
		#CAP_SIZE=2
		
		#plt.clf()
		#plt.xlim(self.Xmin,self.Xmax)
		#plt.ylim(self.Ymin,self.Ymax)
		#plt.xlabel("X direction /km")
		#plt.ylabel("Y direction /km")
		#plt.title('Martime Search and Rescue: '+self.alg_name)
		#marker_cls=['o','+','v','x']
		#color_cls=['y','b','g','r']
		#label_cls=['UAV_Search','UAV_Rotue','USV_Rescue','Person_In_Water']

		To_Plot_OB_X=[]
		To_Plot_OB_Y=[]
		To_Plot_RT_X=[]
		To_Plot_RT_Y=[]
		To_Plot_RS_X=[]
		To_Plot_RS_Y=[]
		To_Plot_TG_X=[]
		To_Plot_TG_Y=[]


		dataGUID=[]
		dataX=[]
		dataY=[]
		dataCLS=[]
		dataSaved=[]

		for p in self.players:
			dataGUID.append(p.guid)
			dataX.append(p.x)
			dataY.append(p.y)
			dataCLS.append(p.cls)
			dataSaved.append(0)
			if(p.cls==ENUM_CLS_OB):
				To_Plot_OB_X.append(p.x)
				To_Plot_OB_Y.append(p.y)
			elif(p.cls==ENUM_CLS_RT):
				To_Plot_RT_X.append(p.x)
				To_Plot_RT_Y.append(p.y)
			elif(p.cls==ENUM_CLS_RS):
				To_Plot_RS_X.append(p.x)
				To_Plot_RS_Y.append(p.y)
			else:
				pass

		for n in self.npcs:
			dataGUID.append(n.guid)
			dataX.append(n.x)
			dataY.append(n.y)
			dataCLS.append(n.cls)
			dataSaved.append(n.is_saved)
			#color=color_cls[n.cls]
			if(n.is_saved==1):
				color='g'
			else:
				To_Plot_TG_X.append(n.x)
				To_Plot_TG_Y.append(n.y)

		#fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)
		plt.figure()
		plt.scatter(To_Plot_OB_X,To_Plot_OB_Y,marker='o',c='y',label='UAV_Search')
		plt.scatter(To_Plot_RT_X,To_Plot_RT_Y,marker='+',c='b',label='UAV_Rotue')
		plt.scatter(To_Plot_RS_X,To_Plot_RS_Y,marker='v',c='g',label='USV_Rescue')
		plt.scatter(To_Plot_TG_X,To_Plot_TG_Y,marker='x',c='r',label='Person_In_Water')
		plt.legend()
		#plt.xlim(-1,1)
		#plt.ylim(-1,1)
		plt.savefig('img/'+self.alg_name+'/'+str(self.env_steps).zfill(4)+'.jpg')
		plt.close()

		dic={
			"GUID":dataGUID,
			"CLS":dataCLS,
			"X":dataX,
			"Y":dataY,
			"saved":dataSaved,
		}
		df = pd.DataFrame(dic)
		df.to_excel('data/'+self.alg_name+'/traj/'+str(self.env_steps).zfill(4)+'.xlsx')

	def check_outbound(self,p,eval=False):
		if(eval):
			return
		if(p.x>self.Xmax):
			p.x=self.Xmax
			#p.penalty_this_step+=-1
		if(p.x<self.Xmin):
			p.x=self.Xmin
			#p.penalty_this_step+=-1
		if(p.y>self.Ymax):
			p.y=self.Ymax
			#p.penalty_this_step+=-1
		if(p.y<self.Ymin):
			p.y=self.Ymin
			#p.penalty_this_step+=-1

	def update_po_info_bygod(self):
		self.update_by_group(self.players)
	def update_po_info_bylinks(self):		
		#从多个one_hop_link构建信息共享最大组
		#遍历self.links,合并其中有共同元素的列表，合并后每个列表是一个信息最大组
		for info_share_max_group in self.merge_player_lists(self.links):
			self.update_by_group(info_share_max_group)

		'''
		info_share_max_group=[]
		for p in self.players:
			in_group=False
			for lk in self.links:
				if(p in lk):
					in_group=True
					break
			if(in_group):
				info_share_max_group.append(p)
			else:
				for po in p.po_copy:
					po.unupdated+=1
		self.update_by_group(info_share_max_group)
		'''
	def update_by_group(self,info_share_max_group):
		if(len(info_share_max_group)<=1):
			return
		self.po_update_cnt+=len(info_share_max_group)*(len(info_share_max_group)-1)
		#整个信息共享最大组内，共享最新的player和npc的信息,组外的unupdated+=1
		#比较计算组内对场景中每一个player的最新信息
		newest_po_copy_p=[]
		for p in self.players:
			newest=p.po_copy[info_share_max_group[0].guid]
			now_unupdated=newest.unupdated
			for p_g in info_share_max_group:
				if(now_unupdated > p.po_copy[p_g.guid].unupdated):
					newest = p.po_copy[p_g.guid]
			newest_po_copy_p.append(newest)
		#对组内每一个player更新场景中每一个player的最新信息
		for p_i in range(len(self.players)):
			for p_g in info_share_max_group:
				po=self.players[p_i].po_copy[p_g.guid]
				newest=newest_po_copy_p[p_i]
				po.x=newest.x
				po.y=newest.y
				po.vx=newest.vx
				po.vy=newest.vy
				po.unupdated=newest.unupdated
		#比较计算组内对场景中每一个npc的最新信息
		newest_po_copy_n=[]
		for n in self.npcs:
			newest=n.po_copy[info_share_max_group[0].guid]
			now_unupdated=newest.unupdated
			for p_g in info_share_max_group:
				if(now_unupdated > n.po_copy[p_g.guid].unupdated):
					newest = n.po_copy[p_g.guid]
			newest_po_copy_n.append(newest)
		#对组内每一个player更新场景中每一个npc的最新信息
		for n_i in range(len(self.npcs)):
			for p_g in info_share_max_group:
				no=self.npcs[n_i].po_copy[p_g.guid]
				newest=newest_po_copy_n[n_i]
				no.x=newest.x
				no.y=newest.y
				no.is_saved=newest.is_saved
				no.unupdated=newest.unupdated
	
	def merge_player_lists(self,player_lists):
		uf = UnionFind()
		for lst in player_lists:
			if not lst:
				continue
			root = lst[0].guid
			for player in lst:
				uf.union(root, player.guid)  # 合并到同一集合
		
		# 按根节点分组
		groups = {}
		for lst in player_lists:
			for player in lst:
				root = uf.find(player.guid)
				if root not in groups:
					groups[root] = []
				groups[root].append(player)
		
		# 去重并返回合并后的列表
		return [list({p.guid: p for p in group}.values()) for group in groups.values()]


class UnionFind:
	def __init__(self):
		self.parent = {}  # 键: player.uid, 值: 父节点uid
		self.rank = {}	# 记录树的高度
	
	def find(self, x_uid):
		if x_uid not in self.parent:
			self.parent[x_uid] = x_uid
			self.rank[x_uid] = 1
		if self.parent[x_uid] != x_uid:
			# 路径压缩：直接指向根节点
			self.parent[x_uid] = self.find(self.parent[x_uid])
		return self.parent[x_uid]
	
	def union(self, x_uid, y_uid):
		root_x = self.find(x_uid)
		root_y = self.find(y_uid)
		if root_x != root_y:
			# 按秩合并
			if self.rank[root_x] > self.rank[root_y]:
				self.parent[root_y] = root_x
			else:
				self.parent[root_x] = root_y
				if self.rank[root_x] == self.rank[root_y]:
					self.rank[root_y] += 1

