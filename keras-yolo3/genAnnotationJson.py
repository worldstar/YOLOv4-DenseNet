import xml.etree.ElementTree as ET
import os
from os import getcwd
import sys

def _main():
    xmlpath   = sys.argv[1] #"./Data/Annotations/"
    imagePath = sys.argv[2] #"./Dreadautomlfile/test/VSDType2/"#"./Data/JPEGImages/"
    writePath = sys.argv[3] #"./model_data/train.txt"
    fr = open(sys.argv[4],'r') #"model_data/voc_classes.txt"
    # classes = ["bicycle","car","cat","dog","person"]
    classes = fr.read().split("\n")
    fr.close()

    for fileName in os.listdir(imagePath):
        # print(fileName)
        if fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
            fw = open((writePath+fileName.replace(".png", ".txt")), "w")
            # print("readFile:",(imagePath+fileName))
            convertResult = convert_annotation((xmlpath+fileName.replace(".png", ".xml")),classes,imagePath)
            fw.write(convertResult)
            # break
    # fw.close()

def dataUs(infos):
    return infos.split(".")[0]

def convert_annotation(path,classes,imagePath): 
    resultstr = ""
    try: 
        xmlFile = open(path) 
    except:
        xmlFile = open(path,encoding="utf-8")   
    xmlTree = ET.parse(xmlFile)
    xmlRoot = xmlTree.getroot()
    width,height,depth = -1,-1,-1
    hasClass = False
    result = ""
    for xmlObj in xmlRoot.iter('size'):
        width = xmlObj.find('width').text.replace(" ", "").replace("\t", "").replace("\n", "")
        height = xmlObj.find('height').text.replace(" ", "").replace("\t", "").replace("\n", "").replace(" ", "").replace("\t", "").replace("\n", "")
        depth = xmlObj.find('depth').text.replace(" ", "").replace("\t", "").replace("\n", "")
        # print(width,height,depth)
    for xmlObj in xmlRoot.iter('object'):
        name = xmlObj.find('name').text.replace(" ", "").replace("\t", "").replace("\n", "")
        isClass = False
        classNum = -1
        for i in range(0,len(classes),1):
            if(name == classes[i]):
                isClass = True
                hasClass = True
                classNum = i
        if(isClass):
            xmin , ymin , xmax , ymax = -1 , -1 , -1 , -1
            for xmlObj2 in xmlObj.iter('bndbox'):
                xmin = int(dataUs(xmlObj2.find('xmin').text.replace(" ", "").replace("\t", "").replace("\n", "").replace(" ", "").replace("\t", "").replace("\n", "")))
                ymin = int(dataUs(xmlObj2.find('ymin').text.replace(" ", "").replace("\t", "").replace("\n", "")))
                xmax = int(dataUs(xmlObj2.find('xmax').text.replace(" ", "").replace("\t", "").replace("\n", "")))
                ymax = int(dataUs(xmlObj2.find('ymax').text.replace(" ", "").replace("\t", "").replace("\n", "")))
                x = xmin / int(width)
                y = ymin / int(height)
                w = (xmax - xmin) / int(width)
                h = (ymax - ymin) / int(height)
            result += "%d %s %s %s %s"%(classNum,x,y,w,h) + "\n"
    # if(hasClass):
    #     FileName = os.path.basename(path)
    #     FileName = os.path.splitext(FileName)[0]
    #     FileName = FileName+"."+deputyFileName
        # result = "%s%s\n"%((imagePath + FileName),result)
    return result

if __name__ == "__main__":
    _main()