1.Data (資料存放)
	1.Data ->
	1-2.Annotations(放入要訓練的Annotations xml檔案)
	1-2.Annotations2(放入要預測的Annotations xml檔案[mAP計算才需使用])
	1-2.JPEGImages(放入要訓練的Image檔案)
	1-2.JPEGImages2(放入要預測的Image檔案)
	1-2.SegmentationClass(產生預測Image結果圖檔)
2.font (字型)
3.mAPTxt (mAP計算所需檔案-Annotations)
4.mAPTxt_Pre (mAP計算所需檔案-Annotations2)
5.model (存放訓練產生model)
6.model_data (存放訓練所需參數資料)
7.yolo3 (主要計算核心)
-------------------------------------------------------------------
訓練步驟
1.執行 genAnnotationClasses.py 產生model_data/voc_classes 檔案
2.執行 genAnnotationTrainPath.py 產生model_data/train.txt 檔案
3.執行 genKmeans.py 產生model_data/yolo_anchors.txt 檔案
4.執行 train.py
-------------------------------------------------------------------
預測步驟
1.執行 predictionGenMAPTxt_Pre.py 預測並產生檔案至
	Data/SegmentationClass以及mAPTxt_pre
2.執行 genAnnotationMAPTxt.py 產生實際對應的檔案至mAPTxt
-------------------------------------------------------------------
評估mAP
1.將mAPTxt、mAPTxt_pre與Data/JPEGImages2放入../mAPCalculate/input內
2.執行main.py進行評估並產生result資料夾
-------------------------------------------------------------------
純回傳資料
1.執行result.py
2.產生相對應result.csv 
