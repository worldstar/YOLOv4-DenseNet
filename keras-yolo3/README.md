# YOLOv3流程(存放方式、訓練、預測、評估、回傳資料等)

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
1. genAnnotationClasses 參數說明
- Folderpath      檔案路徑 ex: ./Data/Annotations/
- writePath   寫入檔案路徑 ex: ./model_data/
- genAnnotationClasses.py 產生 voc_classes 檔案
```
範例: 
python genAnnotationClasses.py ./Data/Annotations/ ./model_data/
自定義:
python genAnnotationClasses.py <Folderpath> <writePath> 
```

2. genAnnotationTrainPath 參數說明
- path            檔案根目錄          ex: ./Data/Annotations/
- imagePath       產生檔案路徑        ex: ./Data/JPEGImages/
- writePath       寫入檔案路徑        ex: ./model_data/
- voc_classesPath 指定voc_classes位置 ex: model_data/voc_classes.txt
- genAnnotationTrainPath.py 產生 train.txt 檔案(待訓練圖片完整路徑、anchorbox)
```
範例: 
python genAnnotationTrainPath.py ./Data/Annotations/ ./Data/JPEGImages/ ./model_data/ model_data/voc_classes.txt
自定義:
python genAnnotationTrainPath.py <path> <imagePath> <writePath> <voc_classesPath>
```

3. genKmeans 參數說明
- trainpath 訓練檔案路徑 ex: model_data/train.txt
- writePath 寫入檔案路徑 ex: model_data/
- genKmeans.py 產生 yolo_anchors.txt 檔案
```
範例: 
python genKmeans.py model_data/train.txt model_data/ 
自定義:
python genKmeans.py <trainpath> <writePath>

```

4. train 參數說明
- log_dir           產生檔案路徑 ex: model/
- annotation_path   檔案名稱 ex: model_data/train.txt
- classes_path      檔案名稱 ex: model_data/voc_classes.txt
- anchors_path      檔案名稱 ex: model_data/yolo_anchors.txt
- valSplit          ex:0.2
- epoch             ex:400
- batchSize         ex:4
- stepMultiple      ex:1
```
範例: 
python train.py model/ model_data/train.txt  model_data/voc_classes.txt  model_data/yolo_anchors.txt 0.2 400 4 1 
自定義:
python train.py <log_dir> <annotation_path> <classes_path> <anchors_path> <valSplit> <epoch> <batchSize> <stepMultiple> 
```

## 預測

1. 執行 [predictionGenMAPTxt_Pre.py](predictionGenMAPTxt_Pre.py) 預測並產生檔案至Data/SegmentationClass以及mAPTxt_pre

2. 執行 [genAnnotationMAPTxt.py](genAnnotationMAPTxt.py) 產生實際對應的檔案至mAPTxt

## [評估(mAP)](../mAPCalculate)

## 回傳資料

1. 執行result.py

2. 產生相對應result.csv 
