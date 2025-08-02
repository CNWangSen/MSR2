import random
import math
from .const import *

class npc_target():
	def __init__(self,po_is_copy=False):
		self.env=None
		self.cls=ENUM_CLS_TG
		self.guid=0

		self.x=0.0
		self.y=0.0
		self.is_saved=0
		self.unupdated=0

		self.po_copy=[]
		self.po_is_copy=po_is_copy
	def reset(self,reset_pos=True):
		if(reset_pos):
			self.x,self.y=self.env.getRandomCircleXY(0,0.5)
		self.is_saved=0
		self.unupdated=0

		if(len(self.po_copy)==0 and self.po_is_copy==False):
			for i in range(len(self.env.players)):
				self.po_copy.append(npc_target(po_is_copy=True))
		if(reset_pos):
			start_x_po=self.x+(random.random()*2-1)*10/UNIT_SPACE#卫星遥感误差10米
			start_y_po=self.y+(random.random()*2-1)*10/UNIT_SPACE#卫星遥感误差10米
			for n_po in self.po_copy:
				n_po.x=start_x_po
				n_po.y=start_y_po
				n_po.is_saved=0
				n_po.unupdated=0
	def step_s_action(self,s_action,eval=False):
		vx,vy=self.ocean_current_v(self.x,self.y)
		self.x+=vx*self.env.current_amp
		self.y+=vy*self.env.current_amp
		#print(vx)
		for n_po in self.po_copy:
			if(n_po.is_saved==0):
				n_po.unupdated+=1
		self.env.check_outbound(self,eval=eval)
	def ocean_current(self,x,y):
		t=self.env.t
		B=TG_B0+TG_e*math.cos(TG_w*t+TG_theta)
		return 1-math.tanh((y-B*math.cos(TG_k*(x-TG_c*t)))/math.sqrt(1+TG_k*TG_k*B*B*math.pow(math.sin(TG_k*(x-TG_c*t)),2)))
	def ocean_current_v(self,x,y):
		d=1e-8
		cent=self.ocean_current(x,y)
		px=-(self.ocean_current(x+d,y)-cent)/d+random.gauss(0,1)
		py=(self.ocean_current(x,y+d)-cent)/d+random.gauss(0,1)
		vx=px*math.cos(self.env.current_dir)+py*math.sin(self.env.current_dir)
		vy=-px*math.sin(self.env.current_dir)+py*math.cos(self.env.current_dir)
		return vx/1600,vy/1600#1600-0.2m/s