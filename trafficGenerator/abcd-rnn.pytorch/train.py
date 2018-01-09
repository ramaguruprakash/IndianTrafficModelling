#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import json
from tqdm import tqdm

from helpers import *
from model import *
from generate import *
import numpy as np

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)
inp_len = 9

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len-1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def get_traffic_training_set2(chunk_len, batch_size):
	li = file.split(", ")
	li = [x.strip("[]'") for x in li]
	sequences = [x.split(' ') for x in li]
	for i,x in enumerate(sequences):
		x[0] = vehicle_to_alphabet[x[0]]
		sequences[i] = x
	
	sequences_len = len(sequences)
	#print sequences, sequences_len
	inp = torch.LongTensor(batch_size, 2*chunk_len)
	target = torch.LongTensor(batch_size, 2*chunk_len)
	for bi in range(batch_size):
		start_index = random.randint(0, sequences_len-chunk_len-1)
		end_index = start_index + chunk_len + 1
		train_seq = sequences[start_index:end_index]
		train_seq = [x[0]+x[1] for x in train_seq]
		train_seq = ''.join(train_seq)
		print "train_seq" , train_seq
		inp[bi] = char_tensor(train_seq[:-2])
		target[bi] = char_tensor(train_seq[2:])
		
	inp = Variable(inp)
	target = Variable(target)
	if args.cuda:
		inp = inpu.cuda()
		target = target.cuda()
	return inp, target


def couple_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, 2*chunk_len)
    target = torch.LongTensor(batch_size, 2*chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - 2*chunk_len-2)
        end_index = start_index + 2*chunk_len + 2
        chunk = file[start_index:end_index]
	print chunk, start_index, end_index, file_len
        inp[bi] = char_tensor(chunk[:-2])
        target[bi] = char_tensor(chunk[2:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def get_traffic_training_set(chunk_len, batch_size):
	li = file.split("], ")
	li[0] = li[0][1:]
	li[len(li)-1] = li[len(li)-1][:-2]
	li = [x + "]" for x in li]
	sequences = [json.loads(x) for x in li]	
	sequence_len = len(sequence)
	print sequence, sequence_len
	inp = torch.LongTensor(batch_size, chunk_len)
	target = torch.LongTensor(batch_size, chunk_len)
	for bi in range(batch_size):
		start_index = random.randint(0, sequence_len-chunk_len-1)
		end_index = start_index + chunk_len + 1
		train_seq = sequence[start_index:end_index]
		inp[bi] = torch.from_numpy(np.array(train_seq[:-1]))
		tartget[bi] = torch.from_numpy(np.array(train_seq[1:]))
	inp = Variable(inp)
	target = Variable(target)
	if args.cuda:
		inp = inpu.cuda()
		target = target.cuda()
	return inp, target


def train(inp, target):
    #print inp.data.numpy(), target.data.numpy()
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
	#print inp[:,c].data.numpy()
        output, hidden = decoder(inp[:,c], hidden)
	print output.view(args.batch_size, -1).data.numpy().shape, target[:,c].data.numpy().shape
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len ## not sure why it is being divided by 0

def train_couple(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
	#print "training " , inp[:,2*c:2*c+2].data.numpy(), target[:,2*c:2*c+2].data.numpy()
        output, hidden = decoder(inp[:,2*c:2*c+2], hidden)
	#print output[:,n_characters+1:].view(args.batch_size, -1).size(), target[:,2*c+1]
        loss1 = criterion(output[:,:n_characters+1].view(args.batch_size, -1), target[:,2*c])
	loss2 = criterion(output[:,n_characters+1:].view(args.batch_size, -1), target[:,2*c+1])
	#print "losses " , loss1.data.numpy(), loss2.data.numpy()
	loss += loss1 + loss2
    loss.backward()
    decoder_optimizer.step()

    return loss.data[0]/ args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training
print n_characters, args.hidden_size, args.model, args.n_layers
decoder = CharRNN(
    n_characters,
    args.hidden_size,
    2*n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        #loss = train(*random_training_set(args.chunk_len, args.batch_size))
        #loss = train_couple(*couple_training_set(args.chunk_len, args.batch_size))
        loss = train_couple(*get_traffic_training_set2(args.chunk_len, args.batch_size))
        #loss = train(get_traffic_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            #print(generate(decoder, 'fgh', 1, cuda=args.cuda), '\n')


    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
