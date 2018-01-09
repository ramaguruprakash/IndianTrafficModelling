#! /users/guruprakash.r/miniconda2/bin/python

from globalHyperP import *
from train import train
from generate import generate
from dataLoader import getTrainingData

if __name__ == "__main__":
#	data = getTrainingData(1000,10,0,0)
#	print "data is fetched " , len(data)
	lo = train(100, 1000)
#	print "data is fetched " , len(data)
	print lo
	## Now train the network, print the loss at every step.
	## Now generate the network after the loss is reduced.