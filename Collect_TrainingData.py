
from env.const import *
import torch
from MSR_IPPO_Transformer import MSR_IPPO
from MSR_MAPPO_Transformer_GroupReward import MSR_MAPPO_GroupReward
from MSR_HAPPO_Transformer import MSR_HAPPO
import numpy
import pandas as pd
import os
import random
import time
#图分为两种：三种算法的训练阶段和评估阶段
#Collect_TrainingData:训练阶段：固定torchseed、scale=1、不同current下,一轮内 vs 轮数,
#训练J_reward、训练savedcnt、训练搜救小组信息更新次数
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
#100:scale=1
#200:scale=10
#404:scale=3
#402:scale=2
class COLLECT_TrainingData:
	def __init__(self):
		self.algs_name={MSR_IPPO:"IPPO_Transformer",MSR_MAPPO_GroupReward:"MAPPO_Transformer_GroupReward",MSR_HAPPO:"HAPPO_Transformer"}
		self.algs=[MSR_HAPPO]
		self.RUN_NUM_PER_ALG=1
		self.EPOCH_PER_RUN=1000
	def collect_data(self):
		run_id=601
		device="cuda:1"
		for scale in [1]:
			for current in [1]:#,10,5,0,2,7]:
				for a in self.algs:
					for seed in [0]:
						run_id+=1
						savepath="data/"+self.algs_name[a]+"/"+"training_"+str(run_id)+"_scale"+str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
						
						#if(os.path.exists(savepath)):
						#	continue

						torch.manual_seed(seed)
						random.seed(seed)
						
						alg_rewards_this_run=[]
						print(savepath,"--begin--",time.ctime())
						t=a(NUM_EPOCH=self.EPOCH_PER_RUN,collecting_paper_data=True,device=device,
							rs_num=1*scale,rt_num=2*scale,ob_num=3*scale,tg_num=80*scale,RUN_ID=run_id,
							map_scaling=scale**(0.5),current_amp=current)
						#t.load()
						t.do_not_save=False
						for _ in range(10):
							t.train(train_num=100)
							print(savepath,"--done--",time.ctime())
							try:
								os.mkdir("data/"+t.alg_name+"/")
							except:
								pass

							dic={}

							for player_i in range(len(t.wp.env.players)):
								dic["actor_loss_player_"+str(player_i)]=t.paper_data_J_actor_loss_train[player_i]
								dic["critic_loss_player_"+str(player_i)]=t.paper_data_J_critic_loss_train[player_i]
								dic["J_rewards_eval_player_"+str(player_i)]=t.paper_data_J_rewards_eval[player_i]
								dic["J_rewards_train_player_"+str(player_i)]=t.paper_data_J_rewards_train[player_i]
							
							dic["save_cnt_train"]=t.paper_data_save_cnt_train
							dic["po_update_cnt_train"]=t.paper_data_po_update_cnt_train
							dic["ob_update_cnt_train"]=t.paper_data_ob_update_cnt_train
							dic["tg_pos_corrected_train"]=t.paper_data_tg_pos_corrected_train

							dic["T_10_eval"]=t.paper_data_T_10_eval
							dic["T_20_eval"]=t.paper_data_T_20_eval
							dic["T_30_eval"]=t.paper_data_T_30_eval
							dic["T_40_eval"]=t.paper_data_T_40_eval
							dic["T_50_eval"]=t.paper_data_T_50_eval
							dic["T_60_eval"]=t.paper_data_T_60_eval
							dic["T_70_eval"]=t.paper_data_T_70_eval
							dic["T_80_eval"]=t.paper_data_T_80_eval
							dic["T_90_eval"]=t.paper_data_T_90_eval
							dic["T_100_eval"]=t.paper_data_T_100_eval
							dic["Final_Percent_eval"]=t.paper_data_Final_Percent_eval
							dic["po_update_cnt_eval"]=t.paper_data_po_update_cnt_eval
							dic["ob_update_cnt_eval"]=t.paper_data_ob_update_cnt_eval
							dic["tg_pos_corrected_eval"]=t.paper_data_tg_pos_corrected_eval
							for key, value in dic.items():
								if(len(value)!=self.EPOCH_PER_RUN+1):
									print(key)


							df = pd.DataFrame(dic)
							df.to_excel(savepath)

if __name__ == '__main__':
	Sol=COLLECT_TrainingData()
	Sol.collect_data()