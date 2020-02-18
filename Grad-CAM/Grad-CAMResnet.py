from keras.applications import imagenet_utils
from keras.applications.resnet50 import  preprocess_input,ResNet50
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.layers import Input,Dense, Dropout, Flatten
from keras.models import Model
#from imageai.Prediction.ResNet.resnet50 import ResNet50
import keras.backend as K
#import tensorflow as tf
import numpy as np
import keras
import cv2
import sys
import os
from matplotlib import pyplot as plt
from os import walk


image_Paths   = "TestImage/VSDType2/"
model_path    = 'model/ep311-loss0.003.h5'
classes_paths = 'model_data/model_class.json'

def get_class():
    classes_path = os.path.expanduser(classes_paths)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

num_classes = len(get_class())

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def processing_image(img_path):
    #print(img_path)
    # 讀取影像為 PIL 影像
    img = image.load_img(img_path, target_size=(224, 224))
    
    # 轉換 PIL 影像為 nparray
    x = image.img_to_array(img)
    
    # 加上一個 batch size，例如轉換 (224, 224, 3) 為 （1, 224, 224, 3) 
    x = np.expand_dims(x, axis=0)
    
    # 將 RBG 轉換為 BGR，並解減去各通道平均
    x = preprocess_input(x)
    
    return x
def decode_predictions_custom(preds, top=5):
    CLASS_CUSTOM = []
    for x in range(1001):
        CLASS_CUSTOM.append(str(x))
        pass
    #print(CLASS_CUSTOM)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        results.append(result)

    return results

def gradcam(model, x):
    # 取得影像的分類類別
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    
    # 取得影像分類名稱
    pred_class_name = decode_predictions_custom(preds)[0][0][0]
    
    # 預測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer('conv1')
    
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    
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
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    
    return heatmap, pred_class_name

def plot_heatmap(heatmap, img_path, pred_class_name):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    print(heatmap)
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
    
    plt.title(pred_class_name)
    #plt.show()

    #plt.show()

num_classes = 4
training_image_size = 224
optimizer ="adagrad"
image_input = (training_image_size, training_image_size, 3)

# model = ResNet50(include_top=False, weights='imagenet',pooling = 'avg',input_shape=image_input)
# num_classes = 4
# x = model.layers[-1].output
# x = Dense(num_classes, activation='softmax', name='predictions')(x)

# # Create your own model 
# model = Model(input=model.input, output=x) 
# model.summary()
#model.load_weights(model_path) 

# model = ResNet50(include_top=True, weights=None,
#                    input_shape=image_input, classes=num_classes)
model = ResNet50(include_top=True, weights=None,
                   input_shape=image_input, classes=num_classes)
model.load_weights(model_path) 
model.summary()
f = []
for (dirpath, dirnames, filenames) in walk(image_Paths):
    f.extend(filenames)
    break
#print(f)
for i in range(len(f)):
    img = processing_image(image_Paths+f[i])
    model2 = model
    heatmap, pred_class_name = gradcam(model2, img)
    plot_heatmap(heatmap, image_Paths+f[i], pred_class_name)
    if i == 5:
        break
        pass
    #break

plt.show()