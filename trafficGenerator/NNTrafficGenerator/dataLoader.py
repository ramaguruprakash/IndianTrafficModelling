#! /users/guruprakash.r/miniconda2/bin/python
from os import listdir
import sys
from os.path import isfile, join
sys.path.append("../../trafficSimulator/")
from vehicleTypes import VehicleTypes
vehicleTypes = VehicleTypes()


def getSequences(filename, use_classes):
	global vehicleTypes
	fp = open(filename, 'r')
	##will not bother about 
	fp.readline()
	fp.readline()
	roadCoordinates = fp.readline()
	roadCoordinates = [float(x)  for x in (roadCoordinates.split(" ")) ]
	print "Road coordinates", roadCoordinates

	##will not bother about
	numberOfseq = int(fp.readline()[:-1])
	sequences = []
	print "numberOfseq", numberOfseq
	for i in range(numberOfseq):
		length_of_seq = 200
		seq = []
		cl = fp.readline()[:-1] ## Class name read.
		for j in range(length_of_seq):
			line = fp.readline()
			encoding = vehicleTypes.oneHotEncoding(cl)
			line = line.split(" ")
			if use_classes:
				seq.append(encoding + [int(line[1]),int(line[2][:-1])])
			else:
				seq.append([int(line[1]), int(line[2][:-1])])
		sequences.append(seq)
	return sequences

def getMultiVehicleSeq(sequences):
	length_seq = 200
	ret = np.zeros((length_seq, len(sequences)));
	for i in range(length_seq):
		for j, sequence in enumerate(sequences):
			ret[i, j] = sequence[i]
	return ret
	


def getTrainingData(root, numOfSeq, batchSize, use_classes, ret_multi):
	'''
		open the file and read the tracks and return based on the batch size. Look the batch size thing.
	'''
	allSeq = []
	onlyfiles = [f for f in listdir(root) if isfile(join(root, f))]
	for filename in onlyfiles:
		if numOfSeq <= 0:
			return allSeq
		print filename
		seq = getSequences(root+filename, use_classes)
		if (len(seq) > numOfSeq):
			seq = seq[:numOfSeq]
			numOfSeq = 0
		else:
			numOfSeq -= len(seq)
		allSeq += seq
	if ret_multi:
		return getMultiVehicleSeq(allSeq)
	else:
		return allSeq


if __name__ == "__main__":
	getTrainingData("../../trafficSimulator/output/",10,10,True,False)
