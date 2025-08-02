import torch
import numpy

def orthogonal_init(layer, gain=1.0):
	torch.nn.init.orthogonal_(layer.weight, gain=gain)
	torch.nn.init.constant_(layer.bias, 0)
	return layer
#观测无人机的指控中心
class ActorAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_hidden=128
        dq_TG=8
        dk_TG=8
        dv_TG=16

        self.linear_Q_TG=orthogonal_init(torch.nn.Linear(2,dq_TG))
        self.linear_K_TG=orthogonal_init(torch.nn.Linear(2,dk_TG))
        self.linear_V_TG=orthogonal_init(torch.nn.Linear(2,dv_TG))

        self.fc_score_OB=torch.nn.Sequential(
            orthogonal_init(torch.nn.Linear(dv_TG,dim_hidden)),
            torch.nn.Tanh(),
            orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
            torch.nn.Tanh(),
            orthogonal_init(torch.nn.Linear(dim_hidden,10),gain=0.01),
        )

        self.fc_score_RS=torch.nn.Sequential(
            orthogonal_init(torch.nn.Linear(dv_TG,dim_hidden)),
            torch.nn.Tanh(),
            orthogonal_init(torch.nn.Linear(dim_hidden,dim_hidden)),
            torch.nn.Tanh(),
            orthogonal_init(torch.nn.Linear(dim_hidden,10),gain=0.01),
        )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4,eps=1e-5)
    def forward(self,state):
        obs_TG = state[:,:,:]#batsh_size,len_seq,len_features
        Q_TG=self.linear_Q_TG(obs_TG)#batch_size,len_seq,dq
        K_TG=self.linear_K_TG(obs_TG)#batch_size,len_seq,dk
        V_TG=self.linear_V_TG(obs_TG)#batch_size,len_seq,dv
        QK_TG=torch.bmm(Q_TG,K_TG.transpose(1,2))/(K_TG.size(-1)**0.5)#batch_size,len_seq,len_seq 
        QKV_TG=torch.bmm(torch.nn.functional.softmax(QK_TG,dim=2),V_TG)#batch_size,len_seq,dv
        	
        scores_OB = self.fc_score_OB(QKV_TG)#batch_size,len_seq,10
        scores_RS = self.fc_score_RS(QKV_TG)#batch_size,len_seq,10
        return scores_OB,scores_RS
    def get_s_action(self,s_obs,eval=False):
        return self.forward(s_obs)
    def save_checkpoint(self,checkpoint_file_prefix,index):
        torch.save(self.state_dict(),checkpoint_file_prefix+"_actor_AC_"+str(index).zfill(2)+".pth")
    def load_checkpoint(self,checkpoint_file_prefix,index):
        self.load_state_dict(torch.load(checkpoint_file_prefix+"_actor_AC_"+str(index).zfill(2)+".pth"))