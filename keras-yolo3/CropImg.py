import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
from PIL import Image
from os import walk

types = "VSDType1"
Tcrop_image = "D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/ResNetDataSet/TCrop.png"
Ccrop_image = "D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/ResNetDataSet/Crop.png"

image_Paths = "D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/ResNetDataSet/test/{}/"
reimage_Paths = "D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/ResNetDataSet/resize/test/{types}}/"

def main():
	x = 0
	y = 0
	x2 = 650
	y2 = 600

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
		#plt.imshow(m2)
		m2.save(reimage_Paths+f[i].replace(".jpg", ".png"))
		#plt.show()
		# if i == 5:
		# 	break
		# 	pass
main()