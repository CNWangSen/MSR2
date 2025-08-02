import torch
import numpy

def orthogonal_init(layer, gain=1.0):
	torch.nn.init.orthogonal_(layer.weight, gain=gain)
	torch.nn.init.constant_(layer.bias, 0)
	return layer

class ActorUAVob(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128):
		super().__init__()
		self.dim_s_obs=dim_s_obs
		self.fc_statu=torch.nn.Sequential(
			torch.nn.Linear(dim_s_obs,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,dim_hidden),
			torch.nn.ReLU(),
		)
		self.fc_mu=torch.nn.Sequential(
			torch.nn.Linear(dim_hidden,dim_act),
			torch.nn.Tanh(),
		)
		self.fc_std=torch.nn.Sequential(
			torch.nn.Linear(dim_hidden,dim_act),
			torch.nn.Softplus(),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
	def forward(self,s_obs):
		state=self.fc_statu(s_obs)
		mu=self.fc_mu(state)
		std=self.fc_std(state)
		return mu,std
	def get_s_action(self,s_obs):
		s_obs = s_obs.reshape(1,self.dim_s_obs)
		mu,std,=self.forward(s_obs)
		action = torch.tanh(torch.distributions.Normal(mu,std).rsample()).tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))


class ActorUAVrt(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128):
		super().__init__()
		self.dim_s_obs=dim_s_obs
		self.fc_statu=torch.nn.Sequential(
			torch.nn.Linear(dim_s_obs,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,dim_hidden),
			torch.nn.ReLU(),
		)
		self.fc_mu=torch.nn.Sequential(
			torch.nn.Linear(dim_hidden,dim_act),
			torch.nn.Tanh(),
		)
		self.fc_std=torch.nn.Sequential(
			torch.nn.Linear(dim_hidden,dim_act),
			torch.nn.Softplus(),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
	def forward(self,s_obs):
		state=self.fc_statu(s_obs)
		mu=self.fc_mu(state)
		std=self.fc_std(state)
		return mu,std
	def get_s_action(self,s_obs):
		s_obs = s_obs.reshape(1,self.dim_s_obs)
		mu,std = self.forward(s_obs)
		action = torch.distributions.Normal(mu,std).rsample().tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))

class ActorUSVrs(torch.nn.Module):
	def __init__(self,dim_s_obs=2,dim_act=2,dim_hidden=128,using_orthogonal_init=True):
		super().__init__()
		self.dim_s_obs=dim_s_obs
		self.fc_statu=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dim_s_obs,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
			torch.nn.Tanh(),
		)
		self.fc_mu=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act)),
			torch.nn.Tanh(),
		)
		self.fc_std=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_act)),
			torch.nn.Softplus(),
		)
		#if(using_orthogonal_init):
		#	orthogonal_init(self.fc_statu)
		#	orthogonal_init(self.fc_mu)
		#	orthogonal_init(self.fc_std, gain=0.01)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	def forward(self,s_obs):
		state=self.fc_statu(s_obs)
		mu=self.fc_mu(state)#+1.0
		std=self.fc_std(state)#+1.0
		return mu,std
	def get_s_action(self,s_obs,eval=False):
		s_obs = s_obs.reshape(1,self.dim_s_obs)
		mu,std,=self.forward(s_obs)
		#if(eval):
		#	return mu.tolist()[0]

		action = torch.distributions.Normal(mu,std).rsample().tolist()[0]
		return action
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))



class CriticUAVob(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		self.fc_statu=torch.nn.Sequential(
			torch.nn.Linear(dim_state,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,1),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
	def forward(self,state):
		score=self.fc_statu(state)
		return score
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))

class CriticUAVrt(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		self.fc_statu=torch.nn.Sequential(
			torch.nn.Linear(dim_state,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,dim_hidden),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_hidden,1),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
	def forward(self,state):
		score=self.fc_statu(state)
		return score
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))

class CriticUSVrs(torch.nn.Module):
	def __init__(self,dim_state=24,dim_hidden=128):
		super().__init__()
		self.fc_statu=torch.nn.Sequential(
			orthogonal_init(torch.nn.Linear(dim_state,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
			torch.nn.Tanh(),
			orthogonal_init(torch.nn.Linear(dim_hidden,1),gain=0.01),
		)
		self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
	def forward(self,state):
		score=self.fc_statu(state)
		return score
	def save_checkpoint(self,checkpoint_file):
		torch.save(self.state_dict(),checkpoint_file)
	def load_checkpoint(self,checkpoint_file):
		self.load_state_dict(torch.load(checkpoint_file))
