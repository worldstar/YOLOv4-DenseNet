
# -*- coding=utf-8 -*-
"""
Created on 2019-6-19 21:39:53
@author: fangsh.Alex
"""
# import keras
# import cv2
# import numpy as np
# import keras.backend as K
# from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
# from yolo3.utils import letterbox_image

# from keras.layers import Input, Lambda
# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing.image import load_img,img_to_array

# def get_classes(classes_path):
#     '''loads the classes'''
#     with open(classes_path) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names

# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)


# K.set_learning_phase(1) #set learning phase
 
# weight_file_dir = 'model_end/epoch20000_博物館YOLOv3.h5'
# # img_path = '/data/sfang/logo_classify/keras_model/error_analyez/0614/0/09219.png'
# img_path = "Data/JPEGImages/20000010015_I001.jpg"
# classes_path = 'model_data/voc_classes.txt'
# anchors_path = 'model_data/yolo_anchors.txt'
# class_names = get_classes(classes_path)
# num_classes = len(class_names)
# num_anchors = len(get_anchors(anchors_path))
# is_tiny_version = num_anchors==6 # default setting
# try:
#     model = load_model(weight_file_dir, compile=False)
# except:
#     model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
#         if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
#     model.load_weights(weight_file_dir) # make sure model, anchors and classes match

# #model = keras.models.load_model(weight_file_dir)
# image = load_img(img_path,target_size=(224,224))
 
# x = img_to_array(image)
# x = np.expand_dims(x,axis=0)
# x = preprocess_input(x)
# pred = model.predict(x)
# class_idx = np.argmax(pred[0])
# print(model.output[0])
# class_output = model.output[:,class_idx]
# last_conv_layer = model.get_layer("block5_conv3")
 
# grads = K.gradients(class_output,last_conv_layer.output)[0]
# pooled_grads = K.mean(grads,axis=(0,1,2))
# iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
# pooled_grads_value, conv_layer_output_value = iterate([x])
# for i in range(512):
#     conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
 
# heatmap = np.mean(conv_layer_output_value, axis=-1)
# heatmap = np.maximum(heatmap,0)
# heatmap /= np.max(heatmap)
 
# img = cv2.imread(img_path)
# img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
# # img = img_to_array(image)
# heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
# superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
# cv2.imshow('Grad-cam',superimposed_img)
# cv2.waitKey(0)

from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.layers import Input
from keras.models import Sequential,Model,load_model
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.model_densenet import densenet_body
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
import sys
import os
from matplotlib import pyplot as plt
from os import walk

image_Paths = "D:/Ultrasound/mydataset/test/VSDType1/"
model_path   = 'model_end/epoch20000_博物館YOLOv3.h5'
# model_path   = 'logs/002/trained_weights_final.h5'
anchors_paths = 'model_data/yolo_anchors.txt'
classes_paths = 'model_data/voc_classes.txt'

def get_class():
    classes_path = os.path.expanduser(classes_paths)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors():
    anchors_path = os.path.expanduser(anchors_paths)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

num_anchors = len(get_anchors())
num_classes = len(get_class())


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def _compute_gradients(tensor, var_list):
	grads = tf.gradients(tensor, var_list)
	return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def processing_image(img_path):
    # 讀取影像為 PIL 影像
    img = image.load_img(img_path, target_size=(224, 224))
    
    # 轉換 PIL 影像為 nparray
    x = image.img_to_array(img)
    
    # 加上一個 batch size，例如轉換 (224, 224, 3) 為 （1, 224, 224, 3) 
    x = np.expand_dims(x, axis=0)
    
    # 將 RBG 轉換為 BGR，並解減去各通道平均
    x = preprocess_input(x)
    
    return x

# preds = model.predict(x)
# pred_class = np.argmax(preds[0])
# model_output = model.output[:pred_class]
# last_conv = model.get_layer('conv2d_143')
# grads = K.gradients(model_output, last_conv.output)[0]
# pooled_grads = K.sum(grads, axis=(0, 1, 2))

# print(pooled_grads)
# return ;
def gradcam(model, x):
    # 取得影像的分類類別
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    
    # 取得影像分類名稱
    # pred_class_name = imagenet_utils.decode_predictions(preds)[0][0][1]
    
    # 預測分類的輸出向量
    #model.summary()
    # print(model.output[3])
    pred_output = model.output[2][0][0][:,pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer('conv2d_75')
    
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    print(grads)
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    print(model.input)
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    return heatmap, "test"

def plot_heatmap(heatmap, img_path, pred_class_name):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    #print(heatmap)
    # 正規化
    heatmap /= np.max(heatmap)
    
    # 讀取影像
    img = cv2.imread(img_path)
    
    fig, ax = plt.subplots()
    
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    
    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.6)
    
    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    
    #plt.title(pred_class_name)
    
    #plt.show()

# preprocessed_input = load_image(sys.argv[1])
#model = densenet_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
model = yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
model.load_weights(model_path) # make sure model, anchors and classes match
model.summary()
#model = ResNet50(weights='imagenet')

# img = processing_image(sys.argv[1])

# heatmap, pred_class_name = gradcam(model, img)

# plot_heatmap(heatmap, sys.argv[1], pred_class_name)

f = []
for (dirpath, dirnames, filenames) in walk(image_Paths):
    f.extend(filenames)
    break

for i in range(len(f)):
    img = processing_image(image_Paths+f[i])
    heatmap, pred_class_name = gradcam(model, img)
    plot_heatmap(heatmap, image_Paths+f[i], pred_class_name)
    break
plt.show()


# model.load_weights(model_path) 
# #model.summary()
# #model = VGG16(weights='imagenet')

# predictions = model.predict(preprocessed_input)[0][0][0]
# #print(predictions)

# # top_1 = decode_predictions(predictions)[0][0]
# # print('Predicted class:')
# # print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
# predicted_class = np.argmax(predictions)

# cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "conv2d_75")
# #cv2.imwrite("gradcam.jpg", cam)

# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model)
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
