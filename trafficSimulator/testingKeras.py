import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input , Dense, Dropout , BatchNormalization, Activation , Add
from keras import regularizers , optimizers
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , CSVLogger
from keras.initializers import TruncatedNormal
from keras.losses import mean_squared_error as mse
import h5py
import matplotlib.pyplot as plt


# load json and create model
json_file = open('kerasmodels/model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("kerasmodels/model_2.h5")
print("Loaded model from disk")


