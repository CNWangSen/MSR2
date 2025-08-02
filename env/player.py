from .const import *
from .tool import *
import math
import random

#搜救总目标：在无人机电量约束下尽快完成搜救任务
#搜救关键词：协同
#模型设计关注：可扩展性（适应可变无人机数量和落水人员数量）

#阶段一：任意艘船营救任意人数，人员位置已知且固定，运行时算法输入人数可变，船之间无通信：已完成
#阶段二：任意艘船和任意个观测无人机营救任意人数，人员位置位置且可变，运行时算法输入人数可变
#阶段三：引入中继无人机


#观测无人机
#功能：1.为搜救船更新落水人员信息
#观测：1.落水者信息（首要）2.搜救艇信息（次要）3.中继无人机信息（再次要）
#动作：1.运动到合适的位置（首要）2.计算卸载决策（摄像头视频质量，卸载目标）
#目标：1.最速更新落水信息（首要）2.节约自身电量（次要）
#难点：输入空间和模型设计，动态更新落水者的信息，

class player_UAV_ob():
	def __init__(self,po_is_copy=False,rt_num=2,ob_num=3,rs_num=1,tg_num=80):
		self.env=None
		self.dim_s_obs=tg_num*3#[dx,dy,unupdated]
		self.dim_s_act=2#[vx,vy]
		self.dim_s_obs_offload=3
		self.dim_s_act_offload=2
		self.cls=ENUM_CLS_OB
		self.dim_feature=3
		self.guid=0

		self.x=0.0
		self.y=0.0
		self.vx=0.0
		self.vy=0.0
		self.penalty_this_step=0
		self.unupdated=0
		self.is_offloading=False
		self.links=[]
		self.compintense=0

		self.po_copy=[]
		self.po_is_copy=po_is_copy
		self.in_my_focus=[]
		self.img_quality=1
	def reset(self,reset_pos=True):
		if(reset_pos):
			self.x=0.0
			self.y=0.0
			self.vx=0.0
			self.vy=0.0

		if(len(self.po_copy)==0 and self.po_is_copy==False):
			for i in range(len(self.env.players)):
				self.po_copy.append(player_UAV_ob(po_is_copy=True))
		for p_po in self.po_copy:
			p_po.x=self.x
			p_po.y=self.y
			p_po.vx=self.vx
			p_po.vy=self.vy
	def get_s_obs(self,eval=False):
		s_obs=[]

		for n in self.env.npcs:
			if(n in self.in_my_focus):
				n_po = n.po_copy[self.guid]
				if(n_po.is_saved==0):
					s_obs.append(n_po.x-self.x)
					s_obs.append(n_po.y-self.y)
					s_obs.append(20*n_po.unupdated/self.env.max_traj_length)
		return s_obs
	def step_s_action(self,s_action,eval=False):
		self.vx=(s_action[0]*2-1)*OB_V_MAX
		self.vy=(s_action[1]*2-1)*OB_V_MAX
		self.x+=self.vx
		self.y+=self.vy
		self.penalty_this_step+= -0.01
		self.update_ob()

		p_id = self.po_copy[self.guid]
		p_id.x=self.x
		p_id.y=self.y
		p_id.unupdated=0

		self.env.check_outbound(self,eval=eval)
	def update_ob(self):
		for n in self.env.npcs:
			if(t_dis(n,self)<OB_CAP_RANGE):
				if(random.random()<=1.0):
					n_po=n.po_copy[self.guid]
					self.env.tg_pos_corrected+=t_dis(n_po,n)
					n_po.x=n.x#+(random.random()*2-1)*70/UNIT_SPACE#摄像头误差10m
					n_po.y=n.y#+(random.random()*2-1)*70/UNIT_SPACE#摄像头误差10m
					n_po.is_saved=n.is_saved
					self.penalty_this_step+=n_po.unupdated/self.env.max_traj_length
					self.env.ob_update_cnt+=n_po.unupdated
					n_po.unupdated=0
	def reset_offload(self,reset_pos=True):
		pass
	def get_s_obs_offload(self,eval=False):#其他无人机位置
		pass
	def step_s_action_offload(self,s_action,eval=False):#摄像头视频质量，本地计算比例
		pass
	def get_distance_tot(self):
		md = 1000000
		for n in self.env.npcs:
			d = t_dis(self,n)
			if(d<=md):
				md=d
		return md
	def get_energy_tot(self):
		return 0#self.get_energy_flight()+self.get_energy_comm()+self.get_energy_comp()
	def get_comm_links(self):
		self.links=[]
		for p in self.env.players_rt:
			d0 = t_dis(p,self.env.players_rs[0])
			if(d0<RT_ROUT_RANGE_A2G):
				d1 = t_dis(self,p)
				if(d1<RT_ROUT_RANGE_A2A):
					self.links.append(d0+d1)		
	def get_tolerate_tot(self):
		self.get_comm_links()
		return len(self.links)
	def get_energy_flight(self):
		v=math.sqrt(self.vx**2+self.vy**2)
		t1=OB_P0*(1+3*v**2/(OB_U_TIP*OB_U_TIP))
		t2=OB_P1*math.sqrt(math.sqrt(1+v**4/(4*OB_U_ROTOR**4))-v**2/(2*OB_U_ROTOR**2))
		t3=OB_d0*OB_s*OB_rou*OB_A*v**3/2
		return t1+t2+t3
	def get_energy_comm(self):
		if(self.is_offloading==1):
			return min(self.links)
		return 0
	def get_energy_comp(self):
		if(self.is_offloading==1):
			return 0
		self.compintense = t_gauss(2,1)
		return self.compintense

#中继无人机
#功能：1.提供通信链路，保持信息更新（首要）2.为观测无人机提供计算卸载服务（次要）
#观测：1.观测无人机信息（首要）2.其他中继无人机信息（次要）
#动作：1.运动到合适的位置
#目标：1.链路稳定性
#难点：整个体系的腰部，起连接作用


class player_UAV_rt():
	def __init__(self,po_is_copy=False,scale=1,rt_num=2,ob_num=3,rs_num=1,tg_num=80):
		self.env=None
		self.dim_s_obs=((rs_num+rt_num+ob_num)-1)*3
		self.dim_s_act=2
		self.dim_s_obs_offload=3
		self.dim_s_act_offload=2
		self.cls=ENUM_CLS_RT
		self.dim_feature=3
		self.guid=0

		self.x=0.0
		self.y=0.0
		self.vx=0.0
		self.vy=0.0
		self.penalty_this_step=0
		self.unupdated=0

		self.po_copy=[]
		self.po_is_copy=po_is_copy
		self.in_my_focus=[]
	def reset(self,reset_pos=True):
		if(reset_pos):
			self.x=0.0
			self.y=0.0
			self.vx=0.0
			self.vy=0.0

		if(len(self.po_copy)==0 and self.po_is_copy==False):
			for i in range(len(self.env.players)):
				self.po_copy.append(player_UAV_ob(po_is_copy=True))
		for p_po in self.po_copy:
			p_po.x=self.x
			p_po.y=self.y
			p_po.vx=self.vx
			p_po.vy=self.vy
	def reset_offload(self,reset_pos=True):
		pass
	def get_s_obs(self,eval=False):
		s_obs=[]

		for p in self.env.players:
			if(p!=self and p in self.in_my_focus):
				p_po = p.po_copy[self.guid]
				s_obs.append(p_po.x-self.x)
				s_obs.append(p_po.y-self.y)
				#s_obs.append(p_po.is_)评估是否入网
				s_obs.append(20*p_po.unupdated/self.env.max_traj_length)
		return s_obs
	def get_s_obs_offload(self,eval=False):
		pass
	def step_s_action(self,s_action,eval=False):
		self.vx=(s_action[0]*2-1)*OB_V_MAX
		self.vy=(s_action[1]*2-1)*OB_V_MAX
		self.x+=self.vx
		self.y+=self.vy
		self.penalty_this_step+= -0.01
		self.update_rt()

		p_id = self.po_copy[self.guid]
		p_id.x=self.x
		p_id.y=self.y
		p_id.unupdated=0

		self.env.check_outbound(self,eval=eval)
	def step_s_action_offload(self,s_action,eval=False):
		pass
	def update_rt(self):
		link_one_hop=[self]
		for p in self.env.players:
			dis = t_dis(self,p)
			inlink=False
			if(p.cls==ENUM_CLS_OB):
				if(dis<=RT_ROUT_RANGE_A2A):
					inlink=True
			elif(p.cls==ENUM_CLS_RS):
				if(dis<=RT_ROUT_RANGE_A2G):
					inlink=True
			if(inlink):
				link_one_hop.append(p)
			else:
				pass
		self.penalty_this_step+=len(link_one_hop)/(10*len(self.env.players))
		self.env.links.append(link_one_hop)

	def get_distance_tot(self):
		return 0
	def get_energy_tot(self):
		return self.get_energy_flight()+self.get_energy_comm()+self.get_energy_comp()
	def get_tolerate_tot(self):
		return 0
	def get_energy_flight(self):
		v=math.sqrt(self.vx**2+self.vy**2)
		t1=OB_P0*(1+3*v**2/(OB_U_TIP*OB_U_TIP))
		t2=OB_P1*math.sqrt(math.sqrt(1+v**4/(4*OB_U_ROTOR**4))-v**2/(2*OB_U_ROTOR**2))
		t3=OB_d0*OB_s*OB_rou*OB_A*v**3/2
		return t1+t2+t3
	def get_energy_comm(self):
		return 0
	def get_energy_comp(self):
		return 0

#搜救船
#功能：1.营救落水者（主要）2.为观测无人机提供计算卸载服务（次要）
#观测：当前落水人员的已知坐标等信息
#动作：运动到合适的位置
#目标：尽快营救
#难点：结合落水人员信息及信息的可信度规划最优搜救路线


class player_USV_rs():
	def __init__(self,po_is_copy=False,rt_num=2,ob_num=3,rs_num=1,tg_num=80):
		self.env=None
		self.dim_s_obs=tg_num*2#[dx,dy]
		self.dim_s_act=2#[vx,vy]
		self.dim_s_obs_offload=3
		self.dim_s_act_offload=2
		self.cls=ENUM_CLS_RS
		self.dim_feature=2
		self.guid=0

		self.x=0.0
		self.y=0.0
		self.vx=0.0
		self.vy=0.0
		self.penalty_this_step=0
		self.unupdated=0
		
		self.po_copy=[]
		self.po_is_copy=po_is_copy
		self.in_my_focus=[]
	def reset(self,reset_pos=True):
		if(reset_pos):
			self.x=0.0
			self.y=0.0
			self.vx=0.0
			self.vy=0.0

		if(len(self.po_copy)==0 and self.po_is_copy==False):
			for i in range(len(self.env.players)):
				self.po_copy.append(player_UAV_ob(po_is_copy=True))
		for p_po in self.po_copy:
			p_po.x=self.x
			p_po.y=self.y
			p_po.vx=self.vx
			p_po.vy=self.vy
	def get_s_obs(self,eval=False):
		s_obs=[]
		for n in self.env.npcs:
			if(n in self.in_my_focus):
				n_po = n.po_copy[self.guid]
				if(n_po.is_saved==0):
					s_obs.append(n_po.x-self.x)
					s_obs.append(n_po.y-self.y)
		return s_obs
	def step_s_action(self,s_action,eval=False):
		self.vx=(s_action[0]*2-1)*RS_V_MAX
		self.vy=(s_action[1]*2-1)*RS_V_MAX
		self.x+=self.vx
		self.y+=self.vy
		self.penalty_this_step+= -0.01
		self.update_save(eval=eval)

		p_id = self.po_copy[self.guid]
		p_id.x=self.x
		p_id.y=self.y
		p_id.unupdated=0

		self.env.check_outbound(self,eval=eval)
	def update_save(self,eval=False):
		for i in range(len(self.env.npcs)-1,-1,-1):
			if(t_dis(self.env.npcs[i],self)<RS_SAVE_RANGE and self.env.npcs[i].is_saved==0):
				n = self.env.npcs[i]
				n.is_saved=1

				n_po=n.po_copy[self.guid]
				n_po.x=n.x
				n_po.y=n.y
				n_po.is_saved=n.is_saved
				n_po.unupdated=0
				
				if(eval==False and self.env.is_training_offload==False):
					self.env.npcs[i].reset()
				self.penalty_this_step+=1
				self.env.save_cnt+=1
	def reset_offload(self,reset_pos=True):
		pass
	def get_s_obs_offload(self,reset_pos=True):
		pass
	def step_s_action_offload(self,s_action_offload,eval=False):#给路由无人机分配带宽
		pass	
	def get_distance_tot(self):
		td=0
		for n in self.env.npcs:
			td+=t_dis(self,n)
		return td
	def get_energy_tot(self):
		return self.get_energy_flight()+self.get_energy_comm()+self.get_energy_comp()
	def get_tolerate_tot(self):
		return 0
	def get_energy_flight(self):
		return 0
	def get_energy_comm(self):
		return 0
	def get_energy_comp(self):
		return 0