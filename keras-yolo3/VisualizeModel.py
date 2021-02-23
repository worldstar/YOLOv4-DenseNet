"""
Retrain the YOLO_densenet model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.callbacks import LambdaCallback
from keras.layers import Input, Lambda
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import yolo_body, yolo_loss,preprocess_true_boxes,yolo_head
from yolo3.model_yolov4 import yolo_bodyV4,yolov4_loss
from yolo3.utils import get_random_data_with_For_Mosaic,get_random_data_with_Mosaic,get_random_data
from yolo3.model_densenet import densenet_body
from yolo3.model_se_densenet import se_densenet_body
import sys
from distutils.util import strtobool
from keras.utils.vis_utils import plot_model

modelname      = sys.argv[1]

def main():
	model = load_model(modelname,compile=False)
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    main()
