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

4. 執行 [CropImg.py](CropImg.py)

    ```
    import warnings
    warnings.filterwarnings('ignore')
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    from os import walk

    types         = "VSDType1"
    Tcrop_image   = "DataSet/TCrop.png"
    Ccrop_image   = "DataSet/Crop.png"
    image_Paths   = "DataSet/{0}/".format(types)
    reimage_Paths = "DataSet/resize/test/{0}}/".format(types)

    def main():
        x = 0
        y = 0
        f = []
        for (dirpath, dirnames, filenames) in walk(image_Paths):
            f.extend(filenames)
            break
        #print(f)
        for i in range(len(f)):
            if f[i].find("Zm")>=0:
                crop_image = Ccrop_image
            else:
                crop_image = Tcrop_image
            bg = Image.open(image_Paths+f[i])

            #bg = bg.resize((705, 579), Image.ANTIALIAS)

            mask = Image.open(crop_image)
            mask_size = mask.size

            crop = bg.crop((x, y, x + mask_size[0], y + mask_size[1]))
            
            m2 = Image.new('RGBA', mask.size)
            m2.paste(crop, mask=mask)
            m2.save(reimage_Paths+f[i].replace(".jpg", ".png"))
    main()
    ```