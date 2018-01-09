#! /Users/gramaguru/anaconda/bin/python

import cv2
import numpy as np

fp = open("NNTrafficGenerator/generatedSequences")

sequences = fp.read()

fp.close()
img = np.zeros((500,500,3))
sequences = sequences.split("]]]")[:-1]
print "Length ", len(sequences)

for sequence in sequences:
	sequence = sequence.split(", ")
	sequence = [s.strip('[') for s in sequence]
	sequence = [s.strip(']') for s in sequence]
	sequence = [(float(s)) for s in sequence]
	for i in range(0, len(sequence), 2):
		print "(" + str(sequence[i]) + "," + str(sequence[i+1]) + ")", 

	print "\n Next Sequence"
	
