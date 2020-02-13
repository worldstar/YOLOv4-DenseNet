# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

---

## YOLOv3整體流程(存放方式、訓練、預測、評估、回傳資料等)

## 測試環境、存放方式

1. 測試環境
    - Python 3.7.1
    - Keras 2.2.4
    - Keras-gup 2.2.4
    - tensorflow 1.14.0
    - tensorflow-gup 1.14.0

2. Data (資料存放)
  - Annotations(放入要訓練的Annotations xml檔案)
  - Annotations2(放入要預測的Annotations xml檔案[mAP計算才需使用])
  - JPEGImages(放入要訓練的Image檔案)
  - JPEGImages2(放入要預測的Image檔案)
  - SegmentationClass(產生預測Image結果圖檔)

3. font (字型)

4. mAPTxt (mAP計算所需檔案-Annotations)

5. mAPTxt_Pre (mAP計算所需檔案-Annotations2)

6. model (存放訓練產生model)

7. model_data (存放訓練所需參數資料)

8. yolo3 (主要計算核心)

## 訓練

1. 執行 [genAnnotationClasses.py](genAnnotationClasses.py) 產生model_data/voc_classes 檔案

2. 執行 [genAnnotationTrainPath.py](genAnnotationTrainPath.py) 產生model_data/train.txt 檔案

3. 執行 [genKmeans.py](genKmeans.py) 產生model_data/yolo_anchors.txt 檔案

4. 執行 [train.py](train.py)


## 預測

1. 執行 [predictionGenMAPTxt_Pre.py](predictionGenMAPTxt_Pre.py) 預測並產生檔案至Data/SegmentationClass以及mAPTxt_pre

2. 執行 [genAnnotationMAPTxt.py](genAnnotationMAPTxt.py) 產生實際對應的檔案至mAPTxt

## [評估(mAP)](../mAPCalculate)

## 回傳資料

1. 執行result.py

2. 產生相對應result.csv 
