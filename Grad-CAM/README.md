# Grad-CAM 可視化解釋基於梯度定位的深度網絡

## 執行檔
- [x] [Grad-CAMYolov3.py](#grad-camyolov3py)
- [x] [Grad-CAMDensenet.py](#grad-camdensenetpy)
- [ ] [ResNet_CAM.py](#grad-camresnetpy)

## 參數說明

```
image_Paths => 圖片資料夾路徑
model_path  => 模型檔案路徑
```

## Grad-CAMYolov3.py
 - image_Paths   圖檔路徑   ex:TestImage/Normal/
 - model_path    h5檔案路徑 ex:model/xxx.h5
 - anchors_paths 檔案路徑   ex:model_data/yolo_anchors.txt
 - classes_paths 檔案路徑   ex:model_data/voc_classes.txt
```
範例: 
python Grad-CAMYolov3.py TestImage/Normal/ model/xxx.h5 yolo_anchors.txt voc_classes.txt
自定義:
python Grad-CAMYolov3.py <image_Paths> <model_path> <anchors_paths> <classes_paths>
```

## ResNet_CAM.py
 - image_Paths         圖檔路徑   ex:TestImage/Normal/
 - model_path          h5檔案路徑 ex:model/xxx.h5
 - num_classes         種類數量   ex:4
 - training_image_size 圖片大小   ex:224

```
範例: 
python ResNet_CAM.py TestImage/Normal/ model/xxx.h5 4 224
自定義:
python ResNet_CAM.py <image_Paths> <model_path> <num_classes> <training_image_size>
```

## Grad-CAMDensenet.py
```

```
