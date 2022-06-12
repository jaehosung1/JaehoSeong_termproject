import os
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, Dense
import numpy as np

from tensorflow.keras.layers import Conv2D,Flatten, Add, Dense, DepthwiseConv2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Reshape, Dropout, Activation, Input
from tensorflow.keras.layers import ReLU, Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten, Add, Dense,  BatchNormalization, Activation, DepthwiseConv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import models, optimizers
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import imgaug as ia
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import imgaug as ia

from ENAS import EfficientNeuralArchitectureSearch
from src.utils import sgdr_learning_rate, get_random_eraser, MixupGenerator

model = load_model(r"C:\Users\5g\Desktop\enas\4\ENAS-Keras-master\src\cifar_test.h5")
model.summary()