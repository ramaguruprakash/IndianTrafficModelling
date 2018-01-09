#! /users/guruprakash.r/miniconda2/bin/python
from myimports import *
from dataLoader import getTrainingData

class TrafficSequencesDataset(Dataset):

	def __init__(self, root, numberOfseq, batchSize, use_classes, ret_multi, transform = None):
		self.root = root
		self.data = getTrainingData(root, numberOfseq, batchSize, use_classes, ret_multi)
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		if self.transform == 'Normalize':
			#print sample
			sample = [[x[0]/float(200), (x[1]-100)/float(200)] for x in sample]
			#print sample
		sample = np.array(sample, dtype=float)
		sample = torch.from_numpy(sample)
		sample = sample.float()


		return sample

if __name__ == "__main__":
	vehicle_dataset = TrafficSequencesDataset("../../trafficSimulator/output/", 100, 10, False, False, 'Normalize')
	for i in range(len(vehicle_dataset)):
		seq = vehicle_dataset[i]
		#print type(seq), seq.size(), (4*400*100)/1000
	dataloader = DataLoader(vehicle_dataset, batch_size=1, shuffle=False, num_workers=2)

	for i, dat in enumerate(dataloader):
		print "Guru", dat.numpy(), dat.numpy().shape
		if i == 0:
			break
