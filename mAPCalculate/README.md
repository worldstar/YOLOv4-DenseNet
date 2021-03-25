# [原著:mAP (mean Average Precision)](https://github.com/Cartucho/mAP)

1. 使用說明
  - 將檔案放入input裡的JPEGImages2，mAPTxt，mAPTxt_Pre
  - ```
  	input
  	  |----JPEGImages2
  	  |----mAPTxt
  	  |----mAPTxt_Pre
    ```
  - 產生result資料夾及其檔案
  - ```
  	results
       |----logs
             |
           <filename>
  	            |----classes
  	            |----images
  	            |----detection-results-info.png
  	            |----ground-truth-info.png
  	            |----lamr.png
  	            |----mAP.png
  	            |----results.txt
    ```
  - main.py 參數說明
  - log_dir  指定predictionGenMAPTxt_Pre.py產出txt存放資料夾
  - filename 指定mAPTxt_Pre/<filename>路徑
  ```
  範例:
  python main.py -log_dir "logs/ASD/YOLOV3-DENSENET20210309V1/" -filename  "sep1000" 
  自定義
  python main.py -log_dir <log_dir> -filename <filename>
  ```
