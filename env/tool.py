import torch
import numpy

def t_dis(p1,p2):
	distance = ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**(0.5)
	return distance

def t_dis_po(p1,p2):
	distance = ((p1.x-p2.x_po)**2+(p1.y-p2.y_po)**2)**(0.5)
	return distance

def t_gauss(miu,sig):
	return 0

def t_clamp(min_value, max_value, value):
	return max(min_value, min(max_value, value))

def t_l2t(l,is_float=False,dim=1):
	if(is_float):
		return torch.FloatTensor(numpy.array(l)).reshape(-1,dim).to("cpu")
	else:
		return torch.LongTensor(numpy.array(l)).reshape(-1,dim).to("cpu")

def t_jl2tl(jl,is_float=False,dim=[]):
	J_t=[]
	for i in range(len(jl)):
		J_t.append(t_l2t(jl[i],is_float,dim[i]))
	return J_t#.to(device)