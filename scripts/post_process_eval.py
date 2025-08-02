import os
import pandas as pd

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
	seed=[1,2,3,4,5,6,7,8,9,10],current=1,RND=True):
	S=[]
	RescueCount_ALL=[]
	RescueCount_SIC=[]
	RescueCount_RND=[]
	T50_ALL=[]
	T50_SIC=[]
	T50_RND=[]

	RescueCount_ALL_mu=[]
	RescueCount_SIC_mu=[]
	RescueCount_RND_mu=[]
	T50_ALL_mu=[]
	T50_SIC_mu=[]
	T50_RND_mu=[]

	RescueCount_ALL_std=[]
	RescueCount_SIC_std=[]
	RescueCount_RND_std=[]
	T50_ALL_std=[]
	T50_SIC_std=[]
	T50_RND_std=[]

	lsd = len(seed)
	for sd in seed:
		dataSIC_path = "../data/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
		"_seed"+str(sd).zfill(2)+\
		"_current"+str(current).zfill(2)+".xlsx"

		if(not os.path.exists(dataSIC_path)):
			print("not exist "+dataSIC_path)

		dataRND_path = "../data/AC_random/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
		"_seed"+str(sd).zfill(2)+\
		"_current"+str(current).zfill(2)+".xlsx"

		if(not os.path.exists(dataRND_path)):
			print("not exist "+dataRND_path)

		dataSIC = pd.read_excel(dataSIC_path)
		if(RND):
			dataRND = pd.read_excel(dataRND_path)

		if(S==[]):
			for i in range(len(dataSIC["scale"])):
				S.append(dataSIC["scale"][i])
				RescueCount_ALL.append([])
				RescueCount_SIC.append([])
				RescueCount_RND.append([])
				T50_ALL.append([])
				T50_SIC.append([])
				T50_RND.append([])


		for i in range(len(dataSIC["scale"])):
			sc = dataSIC["scale"][i]
			RescueCount_ALL[i].append(dataSIC["T_Rescued_Percent_Before"][i]*100)
			RescueCount_SIC[i].append(dataSIC["T_Rescued_Percent_After"][i]*100)
			if(RND):
				RescueCount_RND[i].append(dataRND["T_Rescued_Percent_After"][i]*100)

			t = dataSIC["T_50_Before"][i]
			if t==0:
				t=2160
			T50_ALL[i].append(t*5.0/3600)
			t = dataSIC["T_50_After"][i]
			if t==0:
				t=2160
			T50_SIC[i].append(t*5.0/3600)
			if(RND):
				t = dataRND["T_50_After"][i]
				if t==0:
					t=2160
				T50_RND[i].append(t*5.0/3600)

	for _ in range(len(S)):
		mu,std = CalMean(RescueCount_ALL[_])
		RescueCount_ALL_mu.append(mu)
		RescueCount_ALL_std.append(std)

		mu,std = CalMean(RescueCount_SIC[_])
		RescueCount_SIC_mu.append(mu)
		RescueCount_SIC_std.append(std)		

		mu,std = CalMean(RescueCount_RND[_])
		RescueCount_RND_mu.append(mu)
		RescueCount_RND_std.append(std)	

		mu,std = CalMean(T50_ALL[_])
		T50_ALL_mu.append(mu)
		T50_ALL_std.append(std)

		mu,std = CalMean(T50_SIC[_])
		T50_SIC_mu.append(mu)
		T50_SIC_std.append(std)

		mu,std = CalMean(T50_RND[_])
		T50_RND_mu.append(mu)
		T50_RND_std.append(std)

	if(RND):
		data = {
			'scale':S,

			'RescueCount_ALL_mu': RescueCount_ALL_mu,
			'RescueCount_RND_mu': RescueCount_RND_mu,
			'RescueCount_SIC_mu': RescueCount_SIC_mu,

			"T50_ALL_mu":T50_ALL_mu,
			"T50_RND_mu":T50_RND_mu,
			"T50_SIC_mu":T50_SIC_mu,

			'RescueCount_ALL_std': RescueCount_ALL_std,
			'RescueCount_RND_std': RescueCount_RND_std,
			'RescueCount_SIC_std': RescueCount_SIC_std,

			"T50_ALL_std":T50_ALL_std,
			"T50_RND_std":T50_RND_std,
			"T50_SIC_std":T50_SIC_std,

		}
	else:
		data = {
			'scale':S,

			'RescueCount_ALL_mu': RescueCount_ALL_mu,
			'RescueCount_SIC_mu': RescueCount_SIC_mu,

			"T50_ALL_mu":T50_ALL_mu,
			"T50_SIC_mu":T50_SIC_mu,

			'RescueCount_ALL_std': RescueCount_ALL_std,
			'RescueCount_SIC_std': RescueCount_SIC_std,

			"T50_ALL_std":T50_ALL_std,
			"T50_SIC_std":T50_SIC_std,
		}		

	df = pd.DataFrame(data)
	df.to_excel("../data_post/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
		"_current"+str(current).zfill(2)+".xlsx")

'''
handle_single_file(alg_name="IPPO_Transformer",run_id=101,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1)

handle_single_file(alg_name="MAPPO_Transformer_GroupReward",run_id=101,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1)

handle_single_file(alg_name="HAPPO_Transformer",run_id=101,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1)

handle_single_file(alg_name="IPPO_Transformer",run_id=402,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1,RND=False)

handle_single_file(alg_name="MAPPO_Transformer_GroupReward",run_id=402,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1,RND=False)

handle_single_file(alg_name="HAPPO_Transformer",run_id=402,
	seed=[1,2,3,4,5,6,7,8,9,10],current=1,RND=False)


'''

def handle_single_file_different_current(alg_name="IPPO_Transformer",run_id=101,
	seed=[1,2,3,4,5,6,7,8,9,10],current=[0,1,2,3,4,5,6,7,8,9,10],scale=3,RND=True):
	C=[]
	RescueCount_ALL=[]
	RescueCount_SIC=[]
	RescueCount_RND=[]
	T50_ALL=[]
	T50_SIC=[]
	T50_RND=[]

	RescueCount_ALL_mu=[]
	RescueCount_SIC_mu=[]
	RescueCount_RND_mu=[]
	T50_ALL_mu=[]
	T50_SIC_mu=[]
	T50_RND_mu=[]

	RescueCount_ALL_std=[]
	RescueCount_SIC_std=[]
	RescueCount_RND_std=[]
	T50_ALL_std=[]
	T50_SIC_std=[]
	T50_RND_std=[]

	lsd = len(seed)
	for sd in seed:
		for vel in current:
			dataSIC_path = "../data/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
			"_seed"+str(sd).zfill(2)+\
			"_current"+str(vel).zfill(2)+".xlsx"

			if(not os.path.exists(dataSIC_path)):
				print("not exist "+dataSIC_path)

			dataRND_path = "../data/AC_random/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
			"_seed"+str(sd).zfill(2)+\
			"_current"+str(vel).zfill(2)+".xlsx"

			if(not os.path.exists(dataRND_path)):
				print("not exist "+dataRND_path)

			dataSIC = pd.read_excel(dataSIC_path)
			if(RND):
				dataRND = pd.read_excel(dataRND_path)

			if(C==[]):
				for i in range(len(current)):
					C.append(current[i]/10)
					RescueCount_ALL.append([])
					RescueCount_SIC.append([])
					RescueCount_RND.append([])
					T50_ALL.append([])
					T50_SIC.append([])
					T50_RND.append([])


			for i in range(len(dataSIC["scale"])):
				if(dataSIC["scale"][i]==scale):
					RescueCount_ALL[current.index(vel)].append(dataSIC["T_Rescued_Percent_Before"][i]*100)
					RescueCount_SIC[current.index(vel)].append(dataSIC["T_Rescued_Percent_After"][i]*100)
					t = dataSIC["T_50_Before"][i]
					if t==0:
						t=2160
					T50_ALL[current.index(vel)].append(t*5.0/3600)
					t = dataSIC["T_50_After"][i]
					if t==0:
						t=2160
					T50_SIC[current.index(vel)].append(t*5.0/3600)


			if(RND):
				for i in range(len(dataRND["scale"])):
					if(dataRND["scale"][i]==scale):
						RescueCount_RND[current.index(vel)].append(dataRND["T_Rescued_Percent_After"][i]*100)
						t = dataRND["T_50_After"][i]
						if t==0:
							t=2160
						T50_RND[current.index(vel)].append(t*5.0/3600)



	for _ in range(len(C)):
		mu,std = CalMean(RescueCount_ALL[_])
		RescueCount_ALL_mu.append(mu)
		RescueCount_ALL_std.append(std)

		mu,std = CalMean(RescueCount_SIC[_])
		RescueCount_SIC_mu.append(mu)
		RescueCount_SIC_std.append(std)		

		if(RND):
			mu,std = CalMean(RescueCount_RND[_])
			RescueCount_RND_mu.append(mu)
			RescueCount_RND_std.append(std)	

		mu,std = CalMean(T50_ALL[_])
		T50_ALL_mu.append(mu)
		T50_ALL_std.append(std)

		mu,std = CalMean(T50_SIC[_])
		T50_SIC_mu.append(mu)
		T50_SIC_std.append(std)

		if(RND):
			mu,std = CalMean(T50_RND[_])
			T50_RND_mu.append(mu)
			T50_RND_std.append(std)

	if(not RND):
		data = {
			'Current':C,

			'RescueCount_ALL_mu': RescueCount_ALL_mu,
			'RescueCount_SIC_mu': RescueCount_SIC_mu,

			"T50_ALL_mu":T50_ALL_mu,
			"T50_SIC_mu":T50_SIC_mu,

			'RescueCount_ALL_std': RescueCount_ALL_std,
			'RescueCount_SIC_std': RescueCount_SIC_std,

			"T50_ALL_std":T50_ALL_std,
			"T50_SIC_std":T50_SIC_std,
		}
	else:
		data = {
			'Current':C,

			'RescueCount_ALL_mu': RescueCount_ALL_mu,
			'RescueCount_RND_mu': RescueCount_RND_mu,
			'RescueCount_SIC_mu': RescueCount_SIC_mu,

			"T50_ALL_mu":T50_ALL_mu,
			"T50_RND_mu":T50_RND_mu,
			"T50_SIC_mu":T50_SIC_mu,

			'RescueCount_ALL_std': RescueCount_ALL_std,
			'RescueCount_RND_std': RescueCount_RND_std,
			'RescueCount_SIC_std': RescueCount_SIC_std,

			"T50_ALL_std":T50_ALL_std,
			"T50_RND_std":T50_RND_std,
			"T50_SIC_std":T50_SIC_std,
		}

	df = pd.DataFrame(data)
	df.to_excel("../data_post/AC_Kmeans/evaling_"+alg_name+"_freezedrunid"+str(run_id)+\
		"_scale"+str(scale).zfill(2)+".xlsx")


handle_single_file_different_current(alg_name="IPPO_Transformer",run_id=101)

handle_single_file_different_current(alg_name="MAPPO_Transformer_GroupReward",run_id=101)

handle_single_file_different_current(alg_name="HAPPO_Transformer",run_id=101)


handle_single_file_different_current(alg_name="IPPO_Transformer",run_id=101,scale=1)

handle_single_file_different_current(alg_name="MAPPO_Transformer_GroupReward",run_id=101,scale=1)

handle_single_file_different_current(alg_name="HAPPO_Transformer",run_id=101,scale=1)


handle_single_file_different_current(alg_name="IPPO_Transformer",run_id=106,scale=1,current=[7],RND=False)

handle_single_file_different_current(alg_name="MAPPO_Transformer_GroupReward",run_id=106,scale=1,current=[7],RND=False)

handle_single_file_different_current(alg_name="HAPPO_Transformer",run_id=106,scale=1,current=[7],RND=False)



handle_single_file_different_current(alg_name="IPPO_Transformer",run_id=103,scale=1,current=[5],RND=False)

handle_single_file_different_current(alg_name="MAPPO_Transformer_GroupReward",run_id=103,scale=1,current=[5],RND=False)

handle_single_file_different_current(alg_name="HAPPO_Transformer",run_id=103,scale=1,current=[5],RND=False)