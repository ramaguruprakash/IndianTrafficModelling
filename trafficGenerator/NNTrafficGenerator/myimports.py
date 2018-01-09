#! /users/guruprakash.r/miniconda2/bin/python

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from random import randint, shuffle
import string
import numpy as np