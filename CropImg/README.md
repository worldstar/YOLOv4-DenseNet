# 圖像遮罩、裁減

## 透過蒙版的方式擷取不規則形狀之圖像

1. 測試環境
    - Python 3.7.1

2. 引用套件
    - matplotlib.pyplot
    - PIL
    - warnings
    - os 

3. 參數注意事項
    ```
    mask_size:為從蒙版寬高而來
    crop = bg.crop((x, y, x + mask_size[0], y + mask_size[1]))
    m2 = Image.new('RGBA', mask.size)
    m2.paste(crop, mask=mask)
    【crop】寬高與【m2】不相同將會出錯需多加留意。
    ```

4. 執行 CropImg.py