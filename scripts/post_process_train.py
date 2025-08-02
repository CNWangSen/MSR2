import os
import pandas as pd
import numpy as np

alg_names=["IPPO_Transformer","HAPPO_Transformer","MAPPO_Transformer_GroupReward"]

def CalMean(x_list):
	if(len(x_list)==0):
		return 0,0
	mu=0
	for x in x_list:
		mu+=x
	mu/=len(x_list)
	std=0
	for x in x_list:
		std+=(x-mu)**2
	std=(std/(len(x_list)-1))**(0.5)
	return mu,1.96*std/(len(x_list)**(0.5))
def handle_single_file(alg_name="IPPO_Transformer",run_id=101,
	scale=1,seed=0,current=1,
	group_reward=False,
	):
	
	data_path = "../data/"+alg_name+"/training_"+str(run_id)+"_scale"+\
	str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+\
	"_current"+str(current).zfill(2)+".xlsx"

	if(not os.path.exists(data_path)):
		return

	data = pd.read_excel(data_path)

	CummulativeReward=[]
	RescueCount=[]

	for i in range(len(data["J_rewards_train_player_0"])):
		cummulative_reward=0
		if(group_reward):
			cummulative_reward=data["J_rewards_train_player_0"][i]
		else:
			for p_i in range(6*scale):
				cummulative_reward+=data["J_rewards_train_player_"+str(p_i)][i]

		CummulativeReward.append(cummulative_reward)
		RescueCount.append(data["save_cnt_train"][i])



	data = {
		'CummulativeReward' : CummulativeReward,
		'RescueCount' : RescueCount,
	}

	df = pd.DataFrame(data)
	df.to_excel(data_path.replace("data","data_post"))


handle_single_file(alg_name="IPPO_Transformer",run_id=101,
	scale=1,seed=0,current=1,
	group_reward=False)

handle_single_file(alg_name="MAPPO_Transformer_GroupReward",run_id=101,
	scale=1,seed=0,current=1,
	group_reward=True)

handle_single_file(alg_name="HAPPO_Transformer",run_id=101,
	scale=1,seed=0,current=1,
	group_reward=True)


handle_single_file(alg_name="IPPO_Transformer",run_id=402,
	scale=2,seed=0,current=1,
	group_reward=False)

handle_single_file(alg_name="MAPPO_Transformer_GroupReward",run_id=402,
	scale=2,seed=0,current=1,
	group_reward=True)

handle_single_file(alg_name="HAPPO_Transformer",run_id=402,
	scale=2,seed=0,current=1,
	group_reward=True)

handle_single_file(alg_name="IPPO_Transformer",run_id=102,
	scale=1,seed=0,current=10,
	group_reward=False)

handle_single_file(alg_name="MAPPO_Transformer_GroupReward",run_id=102,
	scale=1,seed=0,current=10,
	group_reward=True)

handle_single_file(alg_name="HAPPO_Transformer",run_id=102,
	scale=1,seed=0,current=10,
	group_reward=True)



def Smooth(y,window_size=30):
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
def moving_average(rewards, window=10):
	smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
	# 填充前 (window-1) 个位置以保持长度一致
	return np.concatenate([rewards[:window-1], smoothed])

def find_converge_epoch_from_reward(reward_list):
	conv_window = 50
	convergence_epoch = None
	smoothed=Smooth(reward_list)

	target = 0.80 * np.max(smoothed)  # 目标 = 最大奖励的 95%
	for i in range(len(smoothed)):
		if np.all(smoothed[i:i+conv_window] >= target):
			convergence_epoch = i
			break
	return convergence_epoch
def find_converge_epoch(alg_name="IPPO_Transformer",run_id=101,scale=1,seed=0,current=1,group_reward=False):
	data_path="../data/"+alg_name+"/training_"+str(run_id)+"_scale"+str(scale).zfill(2)+"_seed"+str(seed).zfill(2)+"_current"+str(current).zfill(2)+".xlsx"

	if(not os.path.exists(data_path)):
		return -1

	data = pd.read_excel(data_path)
	CummulativeReward=[]

	for i in range(len(data["J_rewards_train_player_0"])):
		cummulative_reward=0
		if(group_reward):
			cummulative_reward=data["J_rewards_train_player_0"][i]
		else:
			for p_i in range(6*scale):
				cummulative_reward+=data["J_rewards_train_player_"+str(p_i)][i]
		CummulativeReward.append(cummulative_reward)
	return find_converge_epoch_from_reward(CummulativeReward)

def find_converge_epochs():
	run_ids=[104,101,105,108,109,103,110,106,102]
	currents=[0,1,2,3,4,5,6,7,10]

	ICs=[]
	MCs=[]
	HCs=[]
	for i in range(len(currents)):
		ICs.append(find_converge_epoch(alg_name="IPPO_Transformer",group_reward=False,run_id=run_ids[i],current=currents[i],scale=1,seed=0))
		MCs.append(find_converge_epoch(alg_name="MAPPO_Transformer_GroupReward",group_reward=True,run_id=run_ids[i],current=currents[i],scale=1,seed=0))
		HCs.append(find_converge_epoch(alg_name="HAPPO_Transformer",group_reward=True,run_id=run_ids[i],current=currents[i],scale=1,seed=0))

	data = {
		"current":currents,
		"IPPO":ICs,
		"MAPPO":MCs,
		"HAPPO":HCs
	}

	df = pd.DataFrame(data)
	df.to_excel("../data_post/TrainingConv.xlsx")

find_converge_epochs()