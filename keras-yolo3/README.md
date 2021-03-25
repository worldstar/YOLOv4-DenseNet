# YOLOv3流程(存放方式、訓練、預測、評估、回傳資料等)

## 測試環境、存放方式

1. 測試環境
    - Python 3.7.1
    - Keras 2.2.4
    - Keras-gup 2.2.4
    - tensorflow 1.15.0
    - tensorflow-gup 1.15.0

2. Data (資料存放)
  - Annotations(放入要訓練的Annotations xml檔案)
  - Annotations2(放入要預測的Annotations xml檔案[mAP計算才需使用])
  - JPEGImages(放入要訓練的Image檔案)
  - JPEGImages2(放入要預測的Image檔案)

3. font (字型)

4. mAPTxt (mAP計算所需檔案-Annotations)

5. mAPTxt_Pre (mAP計算所需檔案-Annotations2)

6. model (存放訓練產生model)

7. model_data (存放訓練所需參數資料)

8. yolo3 (主要計算核心)

## 資料處理
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

2. 將labelImg轉換為labelme使用格式 labelImg(xml) to labelme(json)
執行[Xmltojson.py](Xmltojson.py) read xml , png to json 參數說明
- xml_path  xml路徑
- 在xml_path路徑下產出 json 格式
```
範例: 
python Xmltojson.py "./Data/Annotations/" 
python Xmltojson.py <xml_path>
```

3. 將labelme轉換為labelImg使用格式 labelme(json) to labelImg(xml)
執行[labelme2voc.py](labelme2voc.py) read json , png to xml 參數說明
- input_dir  輸入來源(包含圖檔、json檔案)
- output_dir 輸出位址(結果儲存位置)
- --labels   voc_classes 檔案
```
範例:
python labelme2voc.py <input_dir> <output_dir> --labels <labels.txt>

python labelme2voc.py "./Data/ASDType2/" "./Data/test/" --labels "./model_data/voc_classes.txt"

```
4. YOLOV5訓練時使用以下轉換成可訓練資料集 
執行 [genAnnotationJson.py](genAnnotationJson.py) read xml,png to txt , [json資料x1,y1,x2,y2 已經過 normalized] 參數說明 
- xmlpath   xml路徑
- imagePath 圖檔路徑
- writePath 寫入路徑
- fr        voc_class 路徑
```
範例: 
python genAnnotationJson.py "./Data/Annotations/" "./Dreadautomlfile/test/VSDType2/"#"./Data/JPEGImages/" "./model_data/train.txt" "model_data/voc_classes.txt" 
自定義:
python genAnnotationJson.py <xmlpath> <imagePath> <writePath> <fr> 
```

## 訓練

1. genAnnotationTrainPath 參數說明
- path            檔案根目錄          ex: ./Data/Annotations/
- imagePath       產生檔案路徑        ex: ./Data/JPEGImages/
- writetrainPath  寫入訓練檔案路徑     ex: ./model_data/
- writevalPath    寫入預測檔案路徑     ex: ./model_data/
- voc_classesPath 指定voc_classes位置 ex: model_data/voc_classes.txt
- genAnnotationTrainPath.py 產生 train.txt val.txt 檔案(待訓練圖片完整路徑、anchorbox)
```
範例: 
python genAnnotationTrainPath.py ./Data/Annotations/ ./Data/JPEGImages/ ./model_data/train.txt ./model_data/val.txt ./model_data/test.txt model_data/voc_classes.txt
python genAnnotationTrainPath.py ./Dreadautomlfile/valxml/ ./Dreadautomlfile/val/ ./model_data/val.txt model_data/voc_classes.txt
python genAnnotationTrainPath.py ./Data/AnnotationsASD/ ./Data/JPEGImagesASD/ ./model_data/train.txt ./model_data/val.txt ./model_data/test.txt model_data/voc_classes.txt
自定義:
python genAnnotationTrainPath.py <path> <imagePath> <writetrainPath> <writevalPath> <voc_classesPath>
```

2. genKmeans 參數說明
- trainpath 訓練檔案路徑 ex: model_data/train.txt
- writePath 寫入檔案路徑 ex: model_data/
- genKmeans.py 產生 yolo_anchors.txt 檔案
```
範例: 
python genKmeans.py model_data/train.txt model_data/ 
自定義:
python genKmeans.py <trainpath> <writePath>

```

3. train 參數說明
- annotation_path   檔案名稱 ex: model_data/train.txt
- evaluations_path  檔案名稱 ex: model_data/val.txt
- log_dir           產生檔案路徑 ex: model/
- classes_path      檔案名稱 ex: model_data/voc_classes.txt
- anchors_path      檔案名稱 ex: model_data/yolo_anchors.txt
- loadfile_path     檔案名稱 ex: model (若為''則重新訓練 若不為空則讀取<loadfile_path>模型訓練<epoch>次)
- epoch             ex:400
- batchSize         ex:4
```
範例: 
python train.py model/ model_data/train.txt  model_data/voc_classes.txt  model_data/yolo_anchors.txt 0.2 400 4 1 
自定義:
python train.py <log_dir> <annotation_path> <classes_path> <anchors_path> <valSplit> <epoch> <batchSize> <stepMultiple> 
```

## 預測

1. 執行 [predictionGenMAPTxt_Pre.py](predictionGenMAPTxt_Pre.py) 
預測並產生檔案至<write_dir>資料夾內
- readpath        讀取圖檔     ex:JPEGImages/ (僅包含圖檔格式)
- log_dir         模型路徑     ex:{}.h5
- write_dir       寫入路徑     ex:logs/
- modeltype       框架名稱     ex:YOLOV3,YOLOV3Densenet,YOLOV3SE-Densenet,YOLOV4,YOLOV3-SPP,CSPYOLOV3Densenet,CSPSPPYOLOV3Densenet,CSPYOLOV4Densenet
- filetype        檔案類型     ex:txt,xml (xml產出可直接使用LabelImg開啟<write_dir>進行瀏覽)
```
範例: 
python predictionGenMAPTxt_Pre.py "Data/JPEGImagesASD2/*.png" "logs/ASD/YOLOV3-DENSENET20210309V1/sep1000.h5" "mAPTxt_Pre/logs/ASD/YOLOV3-DENSENET20210309V1/sep1000/" YOLOV3Densenet "xml" 
自定義
python predictionGenMAPTxt_Pre.py <readpath> <log_dir> <write_dir> <modeltype> <filetype>

```


2. 執行 [genAnnotationMAPTxt.py](genAnnotationMAPTxt.py) 產生實際對應的檔案至mAPTxt

## [評估(mAP)](../mAPCalculate)

## 回傳資料

1. 執行result.py

2. 產生相對應result.csv 

