import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scienceplots
import subprocess
import shutil
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 对应6号字
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

DPI=600
COLORS=[
"#7e4909","#0e8585","#830783",
"#e5cc8f","#cce5e5","#e5cce5"
]
COLORS=[
"#4c2d50","#732720","#465c79",
"#6a3f70","#A63D33","#6581A6"
]
COLORS=[
"#c72228","#f98f34","#0c4e9b",
"#f5867f","#ffbc80","#6b98c4"
]
LINESTYLE=[
"-","-","-",
"--","--","--"
]
#FONT={'family':'Times New Roman','size':10}
#plt.style.use(['science','ieee'])
#mpl.rcParams['font.family'] = 'SimHei'
#plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

#图分为两种：三种算法的训练阶段和评估阶段
#Collect_TrainingData:训练阶段：固定torchseed、scale=1、不同current下,一轮内 vs 轮数,
#训练J_reward、训练savedcnt、训练搜救小组信息更新次数、训练每个actor/critic的loss、
#					  alg.paper_data_J_rewards_train=[]
#	   env.save_cnt=0 alg.paper_data_save_cnt=[]
#	   env.po_update_cnt=0 alg.paper_data_po_update_cnt_train=[]
#					   alg.paper_data_J_actor_loss_train=[[]] #len_epochx6,6个actor，6个critic，参数共享或中心化critic的赋值相同
#					   alg.paper_data_J_critic_loss_train=[[]]
#评估J_reward、评估Final_Percentage、评估T_10至T_100、评估搜救小组信息更新次数
#				  alg.paper_data_J_rewards_eval=[] #len_epochx6
#		env.T_10=0 alg.paper_data_T_10_eval=[]
#		env.T_20=0 alg.paper_data_T_20_eval=[]
#		env.T_30=0 alg.paper_data_T_30_eval=[]
#		env.T_40=0 alg.paper_data_T_40_eval=[]
#		env.T_50=0 alg.paper_data_T_50_eval=[]
#		env.T_60=0 alg.paper_data_T_60_eval=[]
#		env.T_70=0 alg.paper_data_T_70_eval=[]
#		env.T_80=0 alg.paper_data_T_80_eval=[]
#		env.T_90=0 alg.paper_data_T_90_eval=[]
#		env.T_100=0 alg.paper_data_T_100_eval=[]
#		env.Final_Percent=0 alg.paper_data_Final_Percent_eval=[]
#	   env.po_update_cnt=0 alg.paper_data_po_update_cnt_eval=[]

#Collect_EvalData:评估阶段：不同seed、current下算法开启AC前后 vs scale
#一轮Final_Percentage、T_10、... T_100、搜救小组信息更新次数

class Draw_Curve:
	def __init__(self):
		self.algs_name=["IPPO_Transformer","MAPPO_Transformer_GroupReward","HAPPO_Transformer"]

	def Fetch_TrainingData(self,alg_name="IPPO_Transformer",
		run_id=101,scale=1,seedrange=[0,1,2,3,4,5,6,7,8,9,10],
		current=1,yname="Final_Percent_eval"):

		if(yname=="cumulative_reward_eval"):
			p_l=[0]
			if alg_name=="IPPO_Transformer":
				p_l = [0,1,2,3,4,5]

			y_total=[]
			for seed in seedrange:
				loadpath="../data/"+alg_name+"/"+"training_"+str(run_id)+"_scale"+str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
				if os.path.exists(loadpath):
					data = pd.read_excel(loadpath)
					y_sum = [0.0 for i in range(1001)]
					for p_i in p_l:
						run_one=data["J_rewards_eval_player_"+str(p_i)]
						for i in range(len(run_one)):
							y_sum[i]+=run_one[i]
					y_total.append(y_sum)
		elif(yname=="cumulative_reward_train"):
			p_l=[0]
			if alg_name=="IPPO_Transformer":
				p_l = [0,1,2,3,4,5]

			y_total=[]
			for seed in seedrange:
				loadpath="../data/"+alg_name+"/"+"training_"+str(run_id)+"_scale"+str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
				if os.path.exists(loadpath):
					data = pd.read_excel(loadpath)
					y_sum = [0.0 for i in range(1001)]
					for p_i in p_l:
						run_one=data["J_rewards_train_player_"+str(p_i)]
						for i in range(len(run_one)):
							y_sum[i]+=run_one[i]
					y_total.append(y_sum)
		else:
			y_total=[]
			for seed in seedrange:
				loadpath="../data/"+alg_name+"/"+"training_"+str(run_id)+"_scale"+str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
				if os.path.exists(loadpath):
					data = pd.read_excel(loadpath)
					run_one=data[yname]
					y=[]
					for i in range(len(run_one)):
						if(int(run_one[i])==0 and "T_" in yname):
						   y.append(2160)
						else:
						   y.append(run_one[i])
					y_total.append(y)
		y_mean =numpy.mean(numpy.array(y_total),axis=0)
		y_std=numpy.std(numpy.array(y_total),axis=0)
		#print(y_total)
		y_min=y_mean-0.95*y_std
		y_max=y_mean+0.95*y_std
		return y_mean,y_min,y_max

	def Smooth(self,y,window_size=30):
		decaystep=10
		beta=(1/2.71828)**(1/decaystep)
		ema=[]

		up=0
		down=0
		for i in range(len(y)):
			up=beta*up+y[i]
			down+=beta**(i+1-1)
			ema.append(up/down)
		return ema
	def Draw_Train(self):
		run_id=101
		for current in [1]:#[],10,5,0,2,7]:
			
			self.Draw_Train_Y(yname="cumulative_reward_train",yname_cn="Training Cumulative Reward",current=current,y_min=0,y_max=1000,run_id=run_id,draw_std=True)
			self.Draw_Train_Y(yname="save_cnt_train",yname_cn="Utility of Rescue USV",current=current,y_min=0,y_max=500,run_id=run_id)
			self.Draw_Train_Y(yname="po_update_cnt_train",yname_cn="Utility of Route UAV",current=current,y_min=0,y_max=60000,run_id=run_id)
			self.Draw_Train_Y(yname="ob_update_cnt_train",yname_cn="Utility of Search UAV",current=current,y_min=0,y_max=60000,run_id=run_id)
			self.Draw_Train_Y(yname="tg_pos_corrected_train",yname_cn="Target Position Error",current=current,y_min=0,y_max=60000,run_id=run_id)
			#self.Draw_Train_Y(yname="cumulative_reward_eval",yname_cn="Eval Cumulative Reward",current=current,y_min=-100,y_max=300,run_id=run_id,draw_std=True)
			#self.Draw_Train_Y(yname="T_10_eval",yname_cn="10% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_20_eval",yname_cn="20% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_30_eval",yname_cn="30% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_40_eval",yname_cn="40% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_50_eval",yname_cn="50% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_60_eval",yname_cn="60% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_70_eval",yname_cn="70% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_80_eval",yname_cn="80% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_90_eval",yname_cn="90% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="T_100_eval",yname_cn="100% Rescued Step",current=current,y_min=0,y_max=2160,run_id=run_id)
			#self.Draw_Train_Y(yname="Final_Percent_eval",yname_cn="Eval Rescue Rate",current=current,y_min=0,y_max=1,run_id=run_id)
			#self.Draw_Train_Y(yname="po_update_cnt_eval",yname_cn="Eval Infomation Update Frequency",current=current,y_min=0,y_max=60000,run_id=run_id)
			run_id+=1
	def Draw_Train_Y(self,yname="Final_Percent_eval",yname_cn="一轮营救率",current=1,y_min=0,y_max=1000,run_id=101,draw_std=False):
		plt.clf()
		figname="../fig/TRAIN/"+yname+"_current"+str(current).zfill(2)+".svg"
		title=""#海面平均流速"+str((current/10))+"米/秒时训练过程"+yname_cn
		x=numpy.array([i for i in range(1001)])
		IPPO_mean,IPPO_min,IPPO_max = self.Fetch_TrainingData(alg_name="IPPO_Transformer",run_id=run_id,yname=yname,current=current)
		MAPPO_mean,MAPPO_min,MAPPO_max = self.Fetch_TrainingData(alg_name="MAPPO_Transformer_GroupReward",run_id=run_id,yname=yname,current=current)
		HAPPO_mean,HAPPO_min,HAPPO_max = self.Fetch_TrainingData(alg_name="HAPPO_Transformer",run_id=run_id,yname=yname,current=current)

		fig = plt.figure(1,dpi=DPI)

		plt.xlim(0,1000)
		#plt.ylim(y_min,y_max)
		plt.xlabel("Training Epoch")#,fontdict=FONT)
		plt.ylabel(yname_cn)#,fontdict=FONT)

		#if(draw_std):
		plt.plot(x, self.Smooth(IPPO_mean), label="IPPO", color=COLORS[0],linestyle='-',clip_on=False)#dashdot
		plt.plot(x, self.Smooth(MAPPO_mean), label="MAPPO", color=COLORS[1],linestyle='-',clip_on=False)
		plt.plot(x, self.Smooth(HAPPO_mean), label="HAPPO", color=COLORS[2],linestyle='-',clip_on=False)#dotted
			#print(len(self.Smooth(IPPO_max)),len(self.Smooth(IPPO_min)))
			#plt.fill_between(x, self.Smooth(IPPO_max), self.Smooth(IPPO_min), alpha=0.6, facecolor='#7e4909')
			#plt.fill_between(x, self.Smooth(MAPPO_max), self.Smooth(MAPPO_min), alpha=0.6, facecolor='#0e8585')
			#plt.fill_between(x, self.Smooth(HAPPO_max), self.Smooth(HAPPO_min), alpha=0.6, facecolor='#830783')
		#else:
		#	plt.plot(x, self.Smooth(IPPO_mean), label="IPPO", color=COLORS[0],linestyle='-',clip_on=False)
		#	plt.plot(x, self.Smooth(MAPPO_mean), label="MAPPO", color=COLORS[1],linestyle='-',clip_on=False)
		#	plt.plot(x, self.Smooth(HAPPO_mean), label="HAPPO", color=COLORS[2],linestyle='-',clip_on=False)

		#plt.title(title)#,fontdict=FONT)
		leg=plt.legend()

		leg.get_lines()[0].set_linewidth(3)
		leg.get_lines()[1].set_linewidth(3)
		leg.get_lines()[2].set_linewidth(3)
		plt.grid(False)
		plt.savefig(figname)
		plt.close()
		data = {
			'IPPO_mean': self.Smooth(IPPO_mean),
			'MAPPO_mean': self.Smooth(MAPPO_mean),
			'HAPPO_mean': self.Smooth(HAPPO_mean),
		}

		df = pd.DataFrame(data)
		df.to_excel(figname.replace('.svg','.xlsx'))

	def Fetch_EvalData(self,base_alg_name="IPPO_Transformer",enhance_alg_name="AC_Kmeans",
		run_id=101,scale=1,seedrange=[0,1,2,3,4,5,6,7,8,9,10],
		current=1,yname="Final_Percent_eval"):

		y_total=[]
		for seed in seedrange:
			loadpath="../data/"+enhance_alg_name+"/"+"evaling_"+base_alg_name+"_freezedrunid"+str(run_id)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
			if os.path.exists(loadpath):
				data = pd.read_excel(loadpath)
				if(yname in data.keys()):
					run_one=data[yname]
					y=[]
					for i in range(len(run_one)):
						if(int(run_one[i])==0 and "T_" in yname and "T_Rescued_Percent" not in yname):
							y.append(2160)
						else:
							y.append(run_one[i])
					y_total.append(y)
		if(len(y_total)!=0):
			y_mean =numpy.mean(numpy.array(y_total),axis=0)
			y_std=numpy.std(numpy.array(y_total),axis=0)
			y_min=y_mean-0.95*y_std
			y_max=y_mean+0.95*y_std
		else:
			y_mean=[]
		return y_mean
	def Draw_Eval(self):
		for current in [1]:#,5,10,20]:#[0,1,2,3,4,5,6,7,8,9,10,20]:
			self.Draw_Eval_Y(yname="tg_pos_corrected",yname_cn="Target Position Error",current=current,y_min=0,y_max=600000)
			i1b,m1b,h1b,i1a,m1a,h1a=self.Draw_Eval_Y(yname="ob_update_cnt",yname_cn="Utility of Search UAV",current=current,y_min=0,y_max=600000)
			i3b,m3b,h3b,i3a,m3a,h3a=self.Draw_Eval_Y(yname="po_update_cnt",yname_cn="Utility of Route UAV",current=current,y_min=0,y_max=600000)
			i2b,m2b,h2b,i2a,m2a,h2a=self.Draw_Eval_Y(yname="T_Rescued_Percent",yname_cn="Utility of Rescue USV",current=current,y_min=0,y_max=1)
			#self.Draw_Eval_Y(yname="T_10",yname_cn="10% Rescued Step",current=current,y_min=0,y_max=2160)
			self.Draw_Eval_Y(yname="T_20",yname_cn="20% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_30",yname_cn="30% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_40",yname_cn="40% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_50",yname_cn="50% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_60",yname_cn="60% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_70",yname_cn="70% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_80",yname_cn="80% Rescued Step",current=current,y_min=0,y_max=2160)
			#self.Draw_Eval_Y(yname="T_90",yname_cn="90% Rescued Step",current=current,y_min=0,y_max=2160)
			i4b,m4b,h4b,i4a,m4a,h4a=self.Draw_Eval_Y(yname="T_100",yname_cn="100% Rescued Step",current=current,y_min=0,y_max=2160)

			self.Draw_Eval_Objective([i1b,m1b,h1b,i1a,m1a,h1a],[i2b,m2b,h2b,i2a,m2a,h2a],[i3b,m3b,h3b,i3a,m3a,h3a],[i4b,m4b,h4b,i4a,m4a,h4a],current=current)
	def Draw_Eval_Objective(self,o1,o2,o3,o4,current=1):
		plt.clf()
		figname="../fig/EVAL/Optimization_Objective_current"+str(current).zfill(2)+".svg"
		title=""#"海面平均流速"+str((current/10))+"米/秒时"+yname_cn

		x=numpy.array([i for i in [1,2,3,4,5,6,7,8,9,10]])

		IPPO_mean_Before=[0 for i in range(10)]
		IPPO_mean_After=[0 for i in range(10)]

		MAPPO_mean_Before = [0 for i in range(10)]
		MAPPO_mean_After = [0 for i in range(10)]

		HAPPO_mean_Before = [0 for i in range(10)]
		HAPPO_mean_After = [0 for i in range(10)]
#1.75e7,700,5e5,2160->2000,7000,2000,2160#search,rescue,route
		for i in range(10):
			IPPO_mean_Before[i]+=o1[0][i]*2000/17500000
			IPPO_mean_After[i]+=o1[3][i]*2000/17500000
			MAPPO_mean_Before[i]+=o1[1][i]*2000/17500000
			MAPPO_mean_After[i]+=o1[4][i]*2000/17500000
			HAPPO_mean_Before[i]+=o1[2][i]*2000/17500000
			HAPPO_mean_After[i]+=o1[5][i]*2000/17500000

			IPPO_mean_Before[i]+=o2[0][i]*7000/700
			IPPO_mean_After[i]+=o2[3][i]*7000/700
			MAPPO_mean_Before[i]+=o2[1][i]*7000/700
			MAPPO_mean_After[i]+=o2[4][i]*7000/700
			HAPPO_mean_Before[i]+=o2[2][i]*7000/700
			HAPPO_mean_After[i]+=o2[5][i]*7000/700

			IPPO_mean_Before[i]+=o3[0][i]*4000/500000
			IPPO_mean_After[i]+=o3[3][i]*4000/500000
			MAPPO_mean_Before[i]+=o3[1][i]*4000/500000
			MAPPO_mean_After[i]+=o3[4][i]*4000/500000
			HAPPO_mean_Before[i]+=o3[2][i]*4000/500000
			HAPPO_mean_After[i]+=o3[5][i]*4000/500000

			IPPO_mean_Before[i]-=o4[0][i]*2160/2160
			IPPO_mean_After[i]-=o4[3][i]*2160/2160
			MAPPO_mean_Before[i]-=o4[1][i]*2160/2160
			MAPPO_mean_After[i]-=o4[4][i]*2160/2160
			HAPPO_mean_Before[i]-=o4[2][i]*2160/2160
			HAPPO_mean_After[i]-=o4[5][i]*2160/2160
		fig = plt.figure(1,dpi=DPI)
		plt.xlim(1,10)
		#plt.ylim(y_min,y_max)
		plt.xlabel("Rescue Mission Scale")#,fontdict=FONT)
		plt.ylabel("Optimization Objective")#,fontdict=FONT)

		if(len(IPPO_mean_After)!=0):
			plt.plot(x, IPPO_mean_After, label='SIC-IPPO', color=COLORS[0], linestyle=LINESTYLE[0], marker=">",clip_on=False)
			plt.plot(x, MAPPO_mean_After, label='SIC-MAPPO', color=COLORS[1], linestyle=LINESTYLE[1], marker="o",clip_on=False)
			plt.plot(x, HAPPO_mean_After, label='SIC-HAPPO', color=COLORS[2], linestyle=LINESTYLE[2], marker="s",clip_on=False)
						
			plt.plot(x, IPPO_mean_Before, label='IPPO', color=COLORS[3], linestyle=LINESTYLE[3], marker="x",clip_on=False)
			plt.plot(x, MAPPO_mean_Before, label='MAPPO', color=COLORS[4], linestyle=LINESTYLE[4], marker="*",clip_on=False)
			plt.plot(x, HAPPO_mean_Before, label='HAPPO', color=COLORS[5], linestyle=LINESTYLE[5], marker="d",clip_on=False)


		#plt.fill_between(x, y_max, y_min, alpha=0.6, facecolor='#e75840')
		#plt.fill_between(x, z_max, z_min, alpha=0.6, facecolor='#628cee')
		#plt.title(title)
		plt.legend()
		plt.grid(False)
		plt.savefig(figname)
		plt.close()
		data = {
			'IPPO_mean_Before': IPPO_mean_Before,
			'MAPPO_mean_Before': MAPPO_mean_Before,
			'HAPPO_mean_Before': HAPPO_mean_Before,
			'IPPO_mean_After': IPPO_mean_After,
			'MAPPO_mean_After': MAPPO_mean_After,
			'HAPPO_mean_After': HAPPO_mean_After,
		}

		df = pd.DataFrame(data)
		df.to_excel(figname.replace('.svg','.xlsx'))

	def Draw_Eval_Complex(self):
		for scale in [3,7,10]:#[1,2,3,4,5,6,7,8,9,10]:
			self.Draw_Eval_RPvV(scale=scale)
		#for current in [1,5,10]:#[0,1,2,3,4,5,6,7,8,9,10,20]:
		#	self.Draw_Eval_tvT(current=current)


	def Draw_Eval_Y(self,yname="Final_Percent_eval",yname_cn="一轮营救率",current=1,y_min=0,y_max=1000):
		plt.clf()
		figname="../fig/EVAL/"+yname+"_current"+str(current).zfill(2)+".svg"
		title=""#"海面平均流速"+str((current/10))+"米/秒时"+yname_cn

		x=numpy.array([i for i in [1,2,3,4,5,6,7,8,9,10]])

		IPPO_mean_Before=self.Fetch_EvalData(base_alg_name="IPPO_Transformer",yname=yname+"_Before",current=current)
		IPPO_mean_After=self.Fetch_EvalData(base_alg_name="IPPO_Transformer",yname=yname+"_After",current=current)

		MAPPO_mean_Before = self.Fetch_EvalData(base_alg_name="MAPPO_Transformer_GroupReward",yname=yname+"_Before",current=current)
		MAPPO_mean_After = self.Fetch_EvalData(base_alg_name="MAPPO_Transformer_GroupReward",yname=yname+"_After",current=current)

		HAPPO_mean_Before = self.Fetch_EvalData(base_alg_name="HAPPO_Transformer",yname=yname+"_Before",current=current)
		HAPPO_mean_After = self.Fetch_EvalData(base_alg_name="HAPPO_Transformer",yname=yname+"_After",current=current)

		if(yname_cn=="Utility of Rescue USV"):
			for i in range(10):
				IPPO_mean_Before[i]*=(80*(i+1))
				IPPO_mean_After[i]*=(80*(i+1))
				
				MAPPO_mean_Before[i]*=(80*(i+1))
				MAPPO_mean_After[i]*=(80*(i+1))
				
				HAPPO_mean_Before[i]*=(80*(i+1))
				HAPPO_mean_After[i]*=(80*(i+1))

		fig = plt.figure(1,dpi=DPI)
		plt.xlim(1,10)
		#plt.ylim(y_min,y_max)
		plt.xlabel("Rescue Mission Scale")#,fontdict=FONT)
		plt.ylabel(yname_cn)#,fontdict=FONT)

		if(len(IPPO_mean_After)!=0):
			plt.plot(x, IPPO_mean_After, label='SIC-IPPO', color=COLORS[0], linestyle=LINESTYLE[0], marker=">",clip_on=False)
			plt.plot(x, MAPPO_mean_After, label='SIC-MAPPO', color=COLORS[1], linestyle=LINESTYLE[1], marker="o",clip_on=False)
			plt.plot(x, HAPPO_mean_After, label='SIC-HAPPO', color=COLORS[2], linestyle=LINESTYLE[2], marker="s",clip_on=False)
						
			plt.plot(x, IPPO_mean_Before, label='IPPO', color=COLORS[3], linestyle=LINESTYLE[3], marker="x",clip_on=False)
			plt.plot(x, MAPPO_mean_Before, label='MAPPO', color=COLORS[4], linestyle=LINESTYLE[4], marker="*",clip_on=False)
			plt.plot(x, HAPPO_mean_Before, label='HAPPO', color=COLORS[5], linestyle=LINESTYLE[5], marker="d",clip_on=False)


		#plt.fill_between(x, y_max, y_min, alpha=0.6, facecolor='#e75840')
		#plt.fill_between(x, z_max, z_min, alpha=0.6, facecolor='#628cee')
		#plt.title(title)
		plt.legend()
		plt.grid(False)
		plt.savefig(figname)
		plt.close()
		
		data = {
			'IPPO_mean_Before': IPPO_mean_Before,
			'MAPPO_mean_Before': MAPPO_mean_Before,
			'HAPPO_mean_Before': HAPPO_mean_Before,
			'IPPO_mean_After': IPPO_mean_After,
			'MAPPO_mean_After': MAPPO_mean_After,
			'HAPPO_mean_After': HAPPO_mean_After,
		}

		df = pd.DataFrame(data)
		df.to_excel(figname.replace('.svg','.xlsx'))
		return IPPO_mean_Before,MAPPO_mean_Before,HAPPO_mean_Before,IPPO_mean_After,MAPPO_mean_After,HAPPO_mean_After

	def Fetch_Eval_tvT_Data(self,base_alg_name="IPPO_Transformer",enhance_alg_name="AC_Kmeans",
		run_id=101,scale=1,seedrange=[0,1,2,3,4,5,6,7,8,9,10],
		current=1,yname="_Before"):

		y_total=[]
		for seed in seedrange:
			loadpath="../data/"+enhance_alg_name+"/"+"evaling_"+base_alg_name+"_freezedrunid"+str(run_id)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
			if os.path.exists(loadpath):
				data = pd.read_excel(loadpath)
				y=[]
				for n in ["T_10","T_20","T_30","T_40","T_50","T_60","T_70","T_80","T_90","T_100"]:
					num=data[n+yname][scale-1]
					if(num==0):
						num=2160
					y.append(num)
				y_total.append(y)
		y_mean =numpy.mean(numpy.array(y_total),axis=0)
		y_std=numpy.std(numpy.array(y_total),axis=0)
		y_min=y_mean-0.95*y_std
		y_max=y_mean+0.95*y_std
		return y_mean
	def Draw_Eval_tvT(self,current=1,scale=1):
		plt.clf()
		figname="../fig/EVAL/tvT"+"_current"+str(current).zfill(2)+"_scale"+str(scale).zfill(2)+".svg"
		x=numpy.array([10*i for i in [1,2,3,4,5,6,7,8,9,10]])
		title=""#海面平均流速"+str((current/10))+"米/秒时"+"搜救曲线"
		fig = plt.figure(1,dpi=DPI)
		IPPO_mean_Before=self.Fetch_Eval_tvT_Data(base_alg_name="IPPO_Transformer",yname="_Before",current=current,scale=scale)
		IPPO_mean_After=self.Fetch_Eval_tvT_Data(base_alg_name="IPPO_Transformer",yname="_After",current=current,scale=scale)

		MAPPO_mean_Before = self.Fetch_Eval_tvT_Data(base_alg_name="MAPPO_Transformer_GroupReward",yname="_Before",current=current,scale=scale)
		MAPPO_mean_After = self.Fetch_Eval_tvT_Data(base_alg_name="MAPPO_Transformer_GroupReward",yname="_After",current=current,scale=scale)

		HAPPO_mean_Before = self.Fetch_Eval_tvT_Data(base_alg_name="HAPPO_Transformer",yname="_Before",current=current,scale=scale)
		HAPPO_mean_After = self.Fetch_Eval_tvT_Data(base_alg_name="HAPPO_Transformer",yname="_After",current=current,scale=scale)

		plt.xlim(10,100)
		#plt.ylim(0,2160)
		plt.xlabel("Rescued Percent")#,fontdict=FONT)
		plt.ylabel("Step")#,fontdict=FONT)


		plt.plot(x, IPPO_mean_After, label='SIC-IPPO', color=COLORS[0], linestyle=LINESTYLE[0], marker="D",clip_on=False)
		plt.plot(x, MAPPO_mean_After, label='SIC-MAPPO', color=COLORS[1], linestyle=LINESTYLE[1], marker="o",clip_on=False)
		plt.plot(x, HAPPO_mean_After, label='SIC-HAPPO', color=COLORS[2], linestyle=LINESTYLE[2], marker="s",clip_on=False)
					
		plt.plot(x, IPPO_mean_Before, label='IPPO', color=COLORS[3], linestyle=LINESTYLE[3], marker="x",clip_on=False)
		plt.plot(x, MAPPO_mean_Before, label='MAPPO', color=COLORS[4], linestyle=LINESTYLE[4], marker="*",clip_on=False)
		plt.plot(x, HAPPO_mean_Before, label='HAPPO', color=COLORS[5], linestyle=LINESTYLE[5], marker="d",clip_on=False)


		#plt.fill_between(x, y_max, y_min, alpha=0.6, facecolor='#e75840')
		#plt.fill_between(x, z_max, z_min, alpha=0.6, facecolor='#628cee')
		#plt.title(title)
		plt.legend()
		plt.grid(False)
		plt.savefig(figname)
		plt.close()

	def Fetch_Eval_RPvV_Data(self,base_alg_name="IPPO_Transformer",enhance_alg_name="AC_Kmeans",
		run_id=101,scale=1,seedrange=[0,1,2,3,4,5,6,7,8,9,10],yname="_Before"):
		y_total=[]
		for seed in seedrange:
			y=[]
			for current in [0,1,2,3,4,5,6,7,8,9,10]:
				loadpath="../data/"+enhance_alg_name+"/"+"evaling_"+base_alg_name+"_freezedrunid"+str(run_id)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
				if os.path.exists(loadpath):
					data = pd.read_excel(loadpath)
					num=data["T_Rescued_Percent"+yname][scale-1]
					y.append(num)
				else:
					y.append(0)
				y_total.append(y)
		y_mean =numpy.mean(numpy.array(y_total),axis=0)
		y_std=numpy.std(numpy.array(y_total),axis=0)
		y_min=y_mean-0.95*y_std
		y_max=y_mean+0.95*y_std
		return y_mean
	def Draw_Eval_RPvV(self,scale=1):
		plt.clf()
		figname="../fig/EVAL/RPvV"+"_scale"+str(scale).zfill(2)+".svg"
		x=numpy.array([i/10 for i in [0,1,2,3,4,5,6,7,8,9,10]])
		title=""#尺度"+str(scale)+"时"+"搜救成功率vs流速"
		fig = plt.figure(1,dpi=DPI)
		IPPO_mean_Before=self.Fetch_Eval_RPvV_Data(base_alg_name="IPPO_Transformer",yname="_Before",scale=scale)
		IPPO_mean_After=self.Fetch_Eval_RPvV_Data(base_alg_name="IPPO_Transformer",yname="_After",scale=scale)

		MAPPO_mean_Before = self.Fetch_Eval_RPvV_Data(base_alg_name="MAPPO_Transformer_GroupReward",yname="_Before",scale=scale)
		MAPPO_mean_After = self.Fetch_Eval_RPvV_Data(base_alg_name="MAPPO_Transformer_GroupReward",yname="_After",scale=scale)

		HAPPO_mean_Before = self.Fetch_Eval_RPvV_Data(base_alg_name="HAPPO_Transformer",yname="_Before",scale=scale)
		HAPPO_mean_After = self.Fetch_Eval_RPvV_Data(base_alg_name="HAPPO_Transformer",yname="_After",scale=scale)

		plt.xlim(0,1)
		#plt.ylim(0,1)
		plt.xlabel("Ocean Mean Velocity / (m·s$^{-1}$)")#,fontdict=FONT)
		plt.ylabel("Rescued Percent")#,fontdict=FONT)


		plt.plot(x, IPPO_mean_After, label='SIC-IPPO', color=COLORS[0], linestyle=LINESTYLE[0], marker="D",clip_on=False)
		plt.plot(x, MAPPO_mean_After, label='SIC-MAPPO', color=COLORS[1], linestyle=LINESTYLE[1], marker="o",clip_on=False)
		plt.plot(x, HAPPO_mean_After, label='SIC-HAPPO', color=COLORS[2], linestyle=LINESTYLE[2], marker="s",clip_on=False)

		plt.plot(x, IPPO_mean_Before, label='IPPO', color=COLORS[3], linestyle=LINESTYLE[3], marker="x",clip_on=False)
		plt.plot(x, MAPPO_mean_Before, label='MAPPO', color=COLORS[4], linestyle=LINESTYLE[4], marker="*",clip_on=False)
		plt.plot(x, HAPPO_mean_Before, label='HAPPO', color=COLORS[5], linestyle=LINESTYLE[5], marker="d",clip_on=False)


		#plt.fill_between(x, y_max, y_min, alpha=0.6, facecolor='#e75840')
		#plt.fill_between(x, z_max, z_min, alpha=0.6, facecolor='#628cee')
		#plt.title(title)
		plt.legend()
		plt.grid(False)
		plt.savefig(figname)
		plt.close()		

	def draw_func2(self):
		import numpy as np
		from scipy.stats import norm
		import matplotlib.pyplot as plt
		plt.clf()
		ab_pairs = [(0, 0.1), (0.5, 0.1), (1, 0.1), (0, 0.2), (0.5, 0.2),(1,0.2)]

		x = np.linspace(0, 1, 1002)[1:-1]

		for i in range(len(ab_pairs)):
			a,b=ab_pairs[i]
			dist = norm(a, b)
			y = dist.pdf(x)
			plt.plot(x, y, label=r'$\mu=%.1f,\ \sigma=%.1f$' % (a, b), color=COLORS[i])

		# 设置标题
		#plt.title(u'高斯分布')
		# 设置 x,y 轴取值范围
		plt.xlabel("x")
		plt.ylabel("Gauss(x)")
		plt.xlim(0, 1)
		plt.ylim(0, 4)
		plt.legend()
		plt.savefig("../fig/ART/Gauss.svg", format="svg")
		plt.close()

	def draw_func(self):
		import numpy as np
		from scipy.stats import beta
		import matplotlib.pyplot as plt
		plt.clf()
		ab_pairs = [(0.5, 0.5), (5, 1), (1, 1),(1, 3), (2, 2), (2, 5)]

		x = np.linspace(0, 1, 1002)[1:-1]

		for i in range(len(ab_pairs)):
			a,b=ab_pairs[i]
			dist = beta(a, b)
			y = dist.pdf(x)
			plt.plot(x, y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b), color=COLORS[i])

		# 设置标题
		#plt.title(u'贝塔分布')
		# 设置 x,y 轴取值范围
		plt.xlabel("x")
		plt.ylabel("Beta(x)")
		plt.xlim(0, 1)
		plt.ylim(0, 4)
		plt.legend()
		plt.savefig("../fig/ART/Beta.svg", format="svg")
		plt.close()

	def save_as_wmf(self, fig, filename):
		path = os.path.abspath((filename))
		fig.savefig(path+'.png')
		try:
			subprocess.run(['gswin64c', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=wmf', '-r300', '-sOutputFile='+path+'.wmf', path+'.png'])
		finally:
			os.remove(path+'.png')

	def save_as_emf(self,path):
		for f in os.listdir(path):
			if(".svg" in f):
				shutil.copy(path+f,"D:/Tmp/temp.svg")
				cmd="D:/ProgramFiles/Doc/InkScape/bin/inkscape D:/Tmp/temp.svg --export-filename D:/Tmp/temp.emf --export-text-to-path=no"
				p=subprocess.Popen(cmd)
				p.wait()
				shutil.copy("D:/Tmp/temp.emf",path+f.replace(".svg",".emf"))
				os.remove(path+f)
		print("Convert Done! "+path)


Sol=Draw_Curve()
Sol.save_as_emf("../fig/ART/")
Sol.save_as_emf("img/AC_Kmeans/")
Sol.draw_func()
Sol.draw_func2()
Sol.save_as_emf("../fig/ART/")

Sol.Draw_Eval_Complex()
Sol.Draw_Train()
Sol.Draw_Eval()
Sol.save_as_emf("../fig/EVAL/")
Sol.save_as_emf("../fig/TRAIN/")
Sol.save_as_emf("img/IPPO_Transformer/")
Sol.save_as_emf("img/HAPPO_Transformer/")
Sol.save_as_emf("img/MAPPO_Transformer_GroupReward/")

#Sol.draw_CONV()
#Sol.draw_AC()
