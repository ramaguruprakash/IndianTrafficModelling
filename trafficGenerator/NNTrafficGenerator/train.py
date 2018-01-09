#! /users/guruprakash.r/miniconda2/bin/python
from globalHyperP import *
from generate import generate
from validationLoss import validationLoss
import pdb
import os
import shutil
from trafficSequencesDataset import TrafficSequencesDataset
import subprocess

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


def train(nofEpoch, number_of_seq):
	## Defining the global variables.
	global use_cuda
	global net
	global starting_epoch

	## The entire loss function
	lo = []
	chunk_size = 5

	## Loading the training set
	trafficDataset = TrafficSequencesDataset("../../trafficSimulator/output/", number_of_seq, 10, False, False, 'Normalize')
	trafficDataloader = DataLoader(trafficDataset, batch_size=1, shuffle=False, num_workers=1)

	for i in range(starting_epoch, nofEpoch):

		lossInEpoch = 0

		for k, seq in enumerate(trafficDataloader):
			seq = seq.view(seq.size()[1], seq.size()[2])

			for l, chunk_idx in enumerate(range(0, seq.size()[0], chunk_size)):

				h = net.init_hidden()
				if use_cuda:
					h = h.cuda()

				chunk = seq[chunk_idx:chunk_idx+chunk_size]

				xs, zs = chunk[:-1], chunk[1:]
				if use_cuda:
					xs = xs.cuda()
					zs = zs.cuda()

				loss = 0
				net.zero_grad()
				for x, z in zip(xs, zs):

					x = Variable(x, requires_grad=False)
					z = Variable(z)

					y, h = net(x, h)
					loss += criterion(y.view(1, -1), z)
				lossInEpoch += loss.data.cpu()
				h = h.data
				if use_cuda:
					h = h.cuda()
				h = Variable(h, requires_grad=True)
				loss.backward()
				optimizer.step()
			print "lossInEpoch " + str(i) + " seqNo " + str(k) + " " + str(lossInEpoch.numpy().tolist()) 
			lo.append(lossInEpoch.numpy().tolist())

		if use_cuda:
			save_checkpoint({'epoch':i+1, 
						'state_dict':net.cpu().state_dict(),
						'optimizer':optimizer.state_dict(),
						'GPU_usage':subprocess.check_output(['nvidia-smi']),
						'Losses':str(lo)}, False)
			net = net.cuda()
		else:
			save_checkpoint({'epoch':i+1, 
						'state_dict':net.state_dict(),
						'optimizer':optimizer.state_dict(),
						'GPU_usage':subprocess.check_output(['nvidia-smi']),
						'Losses':str(lo)}, False)

		if i % 5 == 0:
			generate()
	return lo
