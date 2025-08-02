import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scienceplots
import subprocess
import shutil
import pandas as pd

import sys
sys.path.append("..")
from env.const import *

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7.5  # 对应6号字
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'
DPI=72#300
COLORS=[
"#c72228","#f98f34","#0c4e9b",
"#f5867f","#ffbc80","#6b98c4"
]
LINESTYLE=[
"-","-","-",
"--","--","--"
]
alg_short={
	"IPPO_Transformer":"IPPO",
	"MAPPO_Transformer_GroupReward":"MAPPO",
	"HAPPO_Transformer":"HAPPO"
}
PHYWIDTH=2.047/5.2*5.2
PHYHEIGHT=1.532/5.2*5.2
LEG_WIDTH=1
LINE_WIDTH=1
MARKER_SIZE=2.5
CAP_SIZE=2

class Draw_Curve():
	def __init__(self):
		pass
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
		self.Draw_Train_Y(yname="CummulativeReward",yname_cn="Cummulative Reward")
		self.Draw_Train_Y(yname="RescueCount",yname_cn="Rescue Count")

		self.Draw_Train_Y(run_id=102,current=10,yname="CummulativeReward",yname_cn="Cummulative Reward")
		self.Draw_Train_Y(run_id=102,current=10,yname="RescueCount",yname_cn="Rescue Count")

		self.Draw_Train_Y(run_id=402,scale=2,yname="CummulativeReward",yname_cn="Cummulative Reward",RNG=100)
		self.Draw_Train_Y(run_id=402,scale=2,yname="RescueCount",yname_cn="Rescue Count",RNG=100)

	def Fetch_TrainingData(self,alg_name="IPPO_Transformer",
		run_id=101,scale=1,seed=0,current=1,yname="CummulativeReward"):
		loadpath="../data_post/"+alg_name+"/training_"+str(run_id)+"_scale"+\
		str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+\
		"_current"+str(current).zfill(2)+".xlsx"
		data = pd.read_excel(loadpath)
		return data[yname]
	def Draw_Train_Y(self,run_id=101,scale=1,yname="CummulativeReward",current=1,yname_cn="Cummulative Reward",RNG=1001):
		plt.clf()
		figname="../fig_postprocess/TRAIN/"+yname+"_current"+str(current).zfill(2)+".svg"
		if(scale!=1):
			figname="../fig_postprocess/TRAIN/"+yname+"_scale"+str(scale)+"_current"+str(current).zfill(2)+".svg"
		title=""#海面平均流速"+str((current/10))+"米/秒时训练过程"+yname_cn
		x=numpy.array([i for i in range(RNG)])
		IPPO_mean= self.Fetch_TrainingData(alg_name="IPPO_Transformer",run_id=run_id,scale=scale,yname=yname,current=current)
		MAPPO_mean= self.Fetch_TrainingData(alg_name="MAPPO_Transformer_GroupReward",run_id=run_id,scale=scale,yname=yname,current=current)
		HAPPO_mean= self.Fetch_TrainingData(alg_name="HAPPO_Transformer",run_id=run_id,scale=scale,yname=yname,current=current)

		fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)

		plt.xlim(0,RNG)
		#plt.ylim(y_min,y_max)
		plt.xlabel("Training Epoch")#,fontdict=FONT)
		plt.ylabel(yname_cn)#,fontdict=FONT)

		#if(draw_std):
		print(len(IPPO_mean[0:RNG]))
		plt.plot(x, self.Smooth(IPPO_mean[0:RNG]), label="IPPO", color=COLORS[0],linestyle='-',clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dashdot
		plt.plot(x, self.Smooth(MAPPO_mean[0:RNG]), label="MAPPO", color=COLORS[1],linestyle='-',clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)
		plt.plot(x, self.Smooth(HAPPO_mean[0:RNG]), label="HAPPO", color=COLORS[2],linestyle='-',clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dotted
		leg=plt.legend()

		leg.get_lines()[0].set_linewidth(LEG_WIDTH)
		leg.get_lines()[1].set_linewidth(LEG_WIDTH)
		leg.get_lines()[2].set_linewidth(LEG_WIDTH)
		plt.grid(False)
		plt.savefig(figname,bbox_inches='tight')
		plt.close()

	def Draw_Cost(self):
		plt.clf()
		data_path="../data_post/TrainingCost.xlsx"
		data = pd.read_excel(data_path)

		figname="../fig_postprocess/Train/TrainingCost.svg"
		title=""
		x=numpy.array([i for i in range(1,7)])
		IPPO = data["IPPO-GM/GB"]
		MAPPO = data["MAPPO-GM/GB"]
		HAPPO = data["HAPPO-GM/GB"]

		fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)

		plt.xlabel("Rescue Mission Scale")#,fontdict=FONT)
		plt.ylabel("Training GPU Memory/GB")#,fontdict=FONT)

		plt.plot(x, IPPO[0:6], label="IPPO", color=COLORS[0],linestyle='-',marker=">",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dashdot
		plt.plot(x, MAPPO[0:6], label="MAPPO", color=COLORS[1],linestyle='-',marker="o",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)
		plt.plot(x, HAPPO[0:6], label="HAPPO", color=COLORS[2],linestyle='-',marker="*",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dotted
		leg=plt.legend()

		leg.get_lines()[0].set_linewidth(LEG_WIDTH)
		leg.get_lines()[1].set_linewidth(LEG_WIDTH)
		leg.get_lines()[2].set_linewidth(LEG_WIDTH)
		plt.grid(False)
		plt.savefig(figname,bbox_inches='tight')
		plt.close()		
	def Draw_Conv(self):
		plt.clf()
		data_path="../data_post/TrainingConv.xlsx"
		data = pd.read_excel(data_path)

		figname="../fig_postprocess/Train/TrainingConv.svg"
		title=""
		x2=data["current"]
		x=[]
		for x_ in x2:
			x.append(int(x_))
		IPPO = data["IPPO"]
		MAPPO = data["MAPPO"]
		HAPPO = data["HAPPO"]

		fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)
		LEN=7
		plt.xlim(0,LEN)
		plt.xlabel("Ocean Velocity Intensity")#"Mean Ocean Velocity/ (m·s$^{-1}$)")#,fontdict=FONT)
		plt.ylabel("Training Converge Epoch")#,fontdict=FONT)

		plt.plot(x[0:LEN+1], IPPO[0:LEN+1], label="IPPO", color=COLORS[0],linestyle='-',marker=">",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dashdot
		plt.plot(x[0:LEN+1], MAPPO[0:LEN+1], label="MAPPO", color=COLORS[1],linestyle='-',marker="o",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)
		plt.plot(x[0:LEN+1], HAPPO[0:LEN+1], label="HAPPO", color=COLORS[2],linestyle='-',marker="*",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dotted
		leg=plt.legend()

		leg.get_lines()[0].set_linewidth(LEG_WIDTH)
		leg.get_lines()[1].set_linewidth(LEG_WIDTH)
		leg.get_lines()[2].set_linewidth(LEG_WIDTH)
		plt.grid(False)
		plt.savefig(figname,bbox_inches='tight')
		plt.close()	

	def Draw_Eval(self):
		self.Draw_Eval_Y(alg_name="IPPO_Transformer",current=1,miny=0,maxy=100)
		self.Draw_Eval_Y(alg_name="MAPPO_Transformer_GroupReward",current=1,miny=0,maxy=100)
		self.Draw_Eval_Y(alg_name="HAPPO_Transformer",current=1,miny=0,maxy=100)

		self.Draw_Eval_Y(alg_name="IPPO_Transformer",current=1,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)
		self.Draw_Eval_Y(alg_name="MAPPO_Transformer_GroupReward",current=1,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)
		self.Draw_Eval_Y(alg_name="HAPPO_Transformer",current=1,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)

		self.Draw_Eval_Y(alg_name="IPPO_Transformer",scale=3,miny=0,maxy=100)
		self.Draw_Eval_Y(alg_name="MAPPO_Transformer_GroupReward",scale=3,miny=0,maxy=100)
		self.Draw_Eval_Y(alg_name="HAPPO_Transformer",scale=3,miny=0,maxy=100)

		self.Draw_Eval_Y(alg_name="IPPO_Transformer",scale=3,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)
		self.Draw_Eval_Y(alg_name="MAPPO_Transformer_GroupReward",scale=3,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)
		self.Draw_Eval_Y(alg_name="HAPPO_Transformer",scale=3,yname_cn="50% Rescued Time Cost/h",yname="T_50",prefix="T50",miny=0,maxy=3)



	def Draw_Eval_Y(self,alg_name="IPPO_Transformer",run_id=101,current=-1,scale=-1,
		yname_cn="Rescue Success Rate/%",yname="RescueRate",prefix="RescueCount",
		miny=-1,maxy=-1):
		plt.clf()
		fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)

		if(current!=-1):
			data_path="../data_post/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
			"_current"+str(current).zfill(2)+".xlsx"
			figname="../fig_postprocess/EVAL/"+alg_name+"_"+yname+"_current"+str(current).zfill(2)+".svg"
			x=numpy.array([i for i in range(1,11)])
			plt.xlim(1,10)
			plt.xlabel("Rescue Mission Scale")#,fontdict=FONT)
		else:
			data_path="../data_post/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
			"_scale"+str(scale).zfill(2)+".xlsx"
			figname="../fig_postprocess/EVAL/"+alg_name+"_"+yname+"_scale"+str(scale).zfill(2)+".svg"
			x=numpy.array([i for i in range(0,11)])
			plt.xlim(0,10)
			plt.xlabel("Ocean Velocity Intensity")#"Mean Ocean Velocity/ (m·s$^{-1}$)")#,fontdict=FONT)

		data = pd.read_excel(data_path)

		
		title=""#海面平均流速"+str((current/10))+"米/秒时训练过程"+yname_cn
		
		ALL = data[prefix+"_ALL_mu"]
		#if(current!=-1):
		RND = data[prefix+"_RND_mu"]
		SIC = data[prefix+"_SIC_mu"]

		ALL_std = data[prefix+"_ALL_std"]
		#if(current!=-1):
		RND_std = data[prefix+"_RND_std"]
		SIC_std = data[prefix+"_SIC_std"]		

		
		#if(miny!=maxy):
		#	plt.ylim(miny,maxy)
		
		plt.ylabel(yname_cn)#,fontdict=FONT)

		#plt.plot(x, ALL, label="ALL-"+alg_short[alg_name], color=COLORS[0],linestyle='-',marker=">",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dashdot
		#plt.plot(x, RND, label="RND-"+alg_short[alg_name], color=COLORS[1],linestyle='-',marker="o",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)
		#plt.plot(x, SIC, label="SIC-"+alg_short[alg_name], color=COLORS[2],linestyle='-',marker="*",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dotted

		plt.errorbar(x, ALL,yerr=ALL_std, label="ALL-"+alg_short[alg_name],capsize=CAP_SIZE, color=COLORS[0],linestyle='-',marker=">",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dashdot
		#if(current!=-1):
		plt.errorbar(x, RND,yerr=RND_std, label="RND-"+alg_short[alg_name],capsize=CAP_SIZE, color=COLORS[1],linestyle='-',marker="o",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)
		plt.errorbar(x, SIC,yerr=SIC_std, label="SIC-"+alg_short[alg_name],capsize=CAP_SIZE, color=COLORS[2],linestyle='-',marker="*",clip_on=False,linewidth=LINE_WIDTH,markersize=MARKER_SIZE)#dotted
		
		#plt.errorbar(x, y, yerr=yerr, fmt='o', label='Error bars', capsize=5)

		leg=plt.legend()

		#leg.get_lines()[0].set_linewidth(LEG_WIDTH)
		#leg.get_lines()[1].set_linewidth(LEG_WIDTH)
		#leg.get_lines()[2].set_linewidth(LEG_WIDTH)
		plt.grid(False)
		plt.savefig(figname,bbox_inches='tight')
		plt.close()

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

	def Draw_Exe(self,alg_name="HAPPO_Transformer",step=1):
		plt.clf()
		fig = plt.figure(figsize=(PHYWIDTH, PHYHEIGHT),dpi=DPI)
		COLOR_BY_CLS=["#f98f34","#0c4e9b","#00bb00","#c72228"]
		LABEL_BY_CLS=["Search UAV","Route UAV","Rescue USV","Rescue Target"]
		data = pd.read_excel("../data/"+alg_name+"/traj/"+str(step).zfill(4)+".xlsx")
		X=data["X"]
		Y=data["Y"]
		CLS=data["CLS"]
		GUID=data["GUID"]
		SAVED=data["saved"]
		CNT=[0,0,0,0]
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		plt.xlabel("Position X / km")
		plt.ylabel("Position Y / km")
		#OB_CAP_RANGE,RS_SAVE_RANGE,RT_ROUT_RANGE_A2G,RT_ROUT_RANGE_A2A
		for i in range(len(data["GUID"])):
			if(SAVED[i]==0):
				if(CNT[CLS[i]]==0 and step==1):
					plt.scatter(X[i],Y[i],c=COLOR_BY_CLS[CLS[i]],s=MARKER_SIZE,label=LABEL_BY_CLS[CLS[i]])
				else:
					plt.scatter(X[i],Y[i],c=COLOR_BY_CLS[CLS[i]],s=MARKER_SIZE)
				CNT[CLS[i]]+=1
		if(step==1):
			plt.legend()
		plt.savefig("../fig_postprocess/EVAL/traj/"+alg_name+"_"+str(step).zfill(4)+".svg",bbox_inches='tight')
Sol=Draw_Curve()
Sol.Draw_Exe(alg_name="HAPPO_Transformer",step=1)
Sol.Draw_Exe(alg_name="HAPPO_Transformer",step=201)
Sol.Draw_Exe(alg_name="HAPPO_Transformer",step=401)
#Sol.Draw_Train()
#Sol.Draw_Eval()
#Sol.Draw_Cost()
#Sol.Draw_Conv()
Sol.save_as_emf("../fig_postprocess/EVAL/traj/")
#Sol.save_as_emf("../fig_postprocess/TRAIN/")
#Sol.save_as_emf("../fig_postprocess/EVAL/")