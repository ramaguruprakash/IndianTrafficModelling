#! /users/guruprakash.r/miniconda2/bin/python
from globalHyperP import *

def generate():
	fp = open("generatedSequences","a");
	global use_cuda
	global net
	h = net.init_hidden()
	if use_cuda:
		h = h.cuda()
	if use_cuda:
		h = h.cuda()
	x = [-1,-1]
	x = np.array(x, dtype=float)
	x = torch.from_numpy(x)
	x = x.float()
	if use_cuda:
		x = x.cuda()
	x = Variable(x, requires_grad=False)
	seq = []
	for i in range(200):
		pt, h = net(x, h)
		seq.append(pt.data.cpu().numpy().tolist())
		x = pt
	fp.write(str(seq))
	fp.close()
	print seq


if __name__=="__main__":
	generate()