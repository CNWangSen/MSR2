from env.const import *
from MSR_IPPO_Transformer import MSR_IPPO
from MSR_MAPPO_Transformer_GroupReward import MSR_MAPPO_GroupReward
from MSR_HAPPO_Transformer import MSR_HAPPO
from AC_Kmeans import AC_Kmeans
from AC_random import AC_random
import torch
import random
import numpy
import os
import pandas as pd
from matplotlib import pyplot as plt

#Collect_EvalData:评估阶段：不同seed、current下算法开启AC前后 vs scale
#一轮Final_Percentage、T_10、... T_100、搜救小组信息更新次数
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

class COLLECT_EvalData:
	def __init__(self):
		self.algs_name={MSR_IPPO:"IPPO_Transformer",MSR_MAPPO_GroupReward:"MAPPO_Transformer_GroupReward",MSR_HAPPO:"HAPPO_Transformer"}
		self.enhance_algs_name={AC_Kmeans:"AC_Kmeans",AC_random:"AC_random"}
		self.enhance_algs=[AC_random]#AC_Kmeans]
		self.freezed=None

		self.algs=[MSR_IPPO]
		
	def collect_data(self):
		for base_alg in self.algs:
			freezed_base_alg_run_id=103#402
			device="cuda:1"
			for enhance_alg in self.enhance_algs:
				for current in [7]:#[0,1,2,3,4,5,6,7,8,9,10]:#[10,20,1,0]:
					for seed in [1,2,3,4,5,6,7,8,9,10]:
						scale_s = []
						Stochastic_Aided_s=[]
						Kmeans_Aided_s=[]
						Stochastic_Kmeans_Aided_s=[]
						T_10_Before=[]
						T_20_Before=[]
						T_30_Before=[]
						T_40_Before=[]
						T_50_Before=[]
						T_60_Before=[]
						T_70_Before=[]
						T_80_Before=[]
						T_90_Before=[]
						T_100_Before=[]
						T_Rescued_Percent_Before=[]
						po_update_cnt_Before=[]
						ob_update_cnt_Before=[]
						tg_pos_corrected_Before=[]
									
						T_10_After=[]
						T_20_After=[]
						T_30_After=[]
						T_40_After=[]
						T_50_After=[]
						T_60_After=[]
						T_70_After=[]
						T_80_After=[]
						T_90_After=[]
						T_100_After=[]
						T_Rescued_Percent_After=[]
						po_update_cnt_After=[]
						ob_update_cnt_After=[]
						tg_pos_corrected_After=[]

						dic={}
						print(self.algs_name[base_alg]+"---"+self.enhance_algs_name[enhance_alg]+"---CURRENT="+str(current)+"---SEED="+str(seed))
						savepath="data/"+self.enhance_algs_name[enhance_alg]+"/"+"evaling_"+self.algs_name[base_alg]+"_freezedrunid"+str(freezed_base_alg_run_id)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"
						print(savepath)
						#if(os.path.exists(savepath)):
						#	continue
						for scale in [3]:
							'''
							print("scale="+str(scale)+" Before")
							S_epr,freezed_before = self.eval(base_alg=base_alg,enhance_alg=None,using_AC=False,
										device=device,RUN_ID=freezed_base_alg_run_id,
										rt_num=2*scale,ob_num=3*scale,rs_num=scale,tg_num=80*scale,run_num=1,seed=seed,
										stochastic_action=True,map_scaling=scale**(0.5),current_amp=current)
							wpenv=freezed_before.wp.env
							T_10_Before.append(wpenv.T_10)
							T_20_Before.append(wpenv.T_20)
							T_30_Before.append(wpenv.T_30)
							T_40_Before.append(wpenv.T_40)
							T_50_Before.append(wpenv.T_50)
							T_60_Before.append(wpenv.T_60)
							T_70_Before.append(wpenv.T_70)
							T_80_Before.append(wpenv.T_80)
							T_90_Before.append(wpenv.T_90)
							T_100_Before.append(wpenv.T_100)
							T_Rescued_Percent_Before.append(wpenv.Final_Percent)
							po_update_cnt_Before.append(wpenv.po_update_cnt)
							ob_update_cnt_Before.append(wpenv.ob_update_cnt)
							tg_pos_corrected_Before.append(wpenv.tg_pos_corrected)
							print(wpenv.Final_Percent,wpenv.po_update_cnt,wpenv.T_10,wpenv.T_20,wpenv.T_30,wpenv.T_40,wpenv.T_50,wpenv.T_60,wpenv.T_70,wpenv.T_80,wpenv.T_90,wpenv.T_100)
							Stochastic_Aided_s.append(S_epr)
							'''
							print("scale="+str(scale)+" After")
							SK_epr,freezed_after = self.eval(base_alg=base_alg,enhance_alg=enhance_alg,using_AC=True,
										device=device,RUN_ID=freezed_base_alg_run_id,
										rt_num=2*scale,ob_num=3*scale,rs_num=scale,tg_num=80*scale,run_num=1,seed=seed,
										stochastic_action=True,map_scaling=scale**(0.5),current_amp=current)
							wpenv=freezed_after.wp.env
							T_10_After.append(wpenv.T_10)
							T_20_After.append(wpenv.T_20)
							T_30_After.append(wpenv.T_30)
							T_40_After.append(wpenv.T_40)
							T_50_After.append(wpenv.T_50)
							T_60_After.append(wpenv.T_60)
							T_70_After.append(wpenv.T_70)
							T_80_After.append(wpenv.T_80)
							T_90_After.append(wpenv.T_90)
							T_100_After.append(wpenv.T_100)
							T_Rescued_Percent_After.append(wpenv.Final_Percent)
							po_update_cnt_After.append(wpenv.po_update_cnt)
							ob_update_cnt_After.append(wpenv.ob_update_cnt)
							tg_pos_corrected_After.append(wpenv.tg_pos_corrected)
							print(wpenv.Final_Percent,wpenv.po_update_cnt,wpenv.T_10,wpenv.T_20,wpenv.T_30,wpenv.T_40,wpenv.T_50,wpenv.T_60,wpenv.T_70,wpenv.T_80,wpenv.T_90,wpenv.T_100)
							
							scale_s.append(scale)
							
							Stochastic_Kmeans_Aided_s.append(SK_epr)

						dic["scale"] = scale_s
						'''
						dic["EpReward_Before"] = Stochastic_Aided_s
						dic["T_10_Before"] = T_10_Before
						dic["T_20_Before"] = T_20_Before
						dic["T_30_Before"] = T_30_Before
						dic["T_40_Before"] = T_40_Before
						dic["T_50_Before"] = T_50_Before
						dic["T_60_Before"] = T_60_Before
						dic["T_70_Before"] = T_70_Before
						dic["T_80_Before"] = T_80_Before
						dic["T_90_Before"] = T_90_Before
						dic["T_100_Before"] = T_100_Before
						dic["T_Rescued_Percent_Before"] = T_Rescued_Percent_Before
						dic["po_update_cnt_Before"] = po_update_cnt_Before
						dic["ob_update_cnt_Before"] = ob_update_cnt_Before
						dic["tg_pos_corrected_Before"] = tg_pos_corrected_Before
						'''
						dic["EpReward_After"] = Stochastic_Kmeans_Aided_s
						dic["T_10_After"] = T_10_After
						dic["T_20_After"] = T_20_After
						dic["T_30_After"] = T_30_After
						dic["T_40_After"] = T_40_After
						dic["T_50_After"] = T_50_After
						dic["T_60_After"] = T_60_After
						dic["T_70_After"] = T_70_After
						dic["T_80_After"] = T_80_After
						dic["T_90_After"] = T_90_After
						dic["T_100_After"] = T_100_After
						dic["T_Rescued_Percent_After"] = T_Rescued_Percent_After
						dic["po_update_cnt_After"] = po_update_cnt_After
						dic["ob_update_cnt_After"] = ob_update_cnt_After
						dic["tg_pos_corrected_After"] = tg_pos_corrected_After

						df = pd.DataFrame(dic)
						try:
							os.mkdir("data/"+t.alg_name)
						except:
							pass
						df.to_excel(savepath)
	def eval(self,base_alg=MSR_IPPO,enhance_alg=AC_Kmeans,
		  using_AC=True,device="cuda:2",
		  RUN_ID=101,rt_num=2,ob_num=3,
		  rs_num=1,tg_num=80,run_num=1,
		  seed=42,stochastic_action=True,
		  map_scaling=1,current_amp=1):
		with torch.no_grad():
			freezed=base_alg(device=device,RUN_ID=RUN_ID,
					rt_num=rt_num,ob_num=ob_num,rs_num=rs_num,tg_num=tg_num,
					stochastic_action=stochastic_action,
					map_scaling=map_scaling,current_amp=current_amp)
			freezed.load()
			rewards_eval=[]
			for _ in range(run_num):
				random.seed(seed)
				torch.manual_seed(seed)
				freezed.wp.reset()
				if(using_AC):
					enhance_alg().Enhance(freezed)
				reward=freezed.get_traj(show=False,eval=True,reset_pos=False)
				rewards_eval.append(reward)
				#print(reward)
			test_result = sum(rewards_eval)/run_num
			#print(test_result)
			return test_result,freezed

if __name__ == '__main__':
	Sol=COLLECT_EvalData()
	Sol.collect_data()