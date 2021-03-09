import xml.etree.ElementTree as ET
import os
from os import getcwd
import sys
import numpy as np

def _main():
    path = sys.argv[1] #"./Data/Annotations/"
    imagePath = sys.argv[2]#"./Data/JPEGImages/"
    writetrainPath = sys.argv[3]#"./model_data/train.txt"
    writevalPath = sys.argv[4]#"./model_data/train.txt"
    lines = []
    deputyFileName = "png"
    # classes = ["bicycle","car","cat","dog","person"]
    fr = open(sys.argv[5] , 'r')#"model_data/voc_classes.txt"
    classes = fr.read().split("\n")
    fr.close()

    fw = open(writetrainPath, "w")
    fw2 = open(writevalPath, "w")

    # with open(path) as f:
    #     lines = f.readlines()

    for fileName in os.listdir(path):
        if fileName in ".gitignore":
            continue
        lines.append((path+fileName))

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val
    print(len(lines))
    print(len(lines[:num_train]))
    print(len(lines[num_train:]))

    # print(lines[num_train:])

    for fileName in lines[:num_train]:
        if fileName in ".gitignore":
            continue
        convertResult = convert_annotation((fileName),classes,imagePath,deputyFileName)
        fw.write(convertResult)

    for fileName in lines[num_train:]:
        if fileName in ".gitignore":
            continue
        convertResult = convert_annotation((fileName),classes,imagePath,deputyFileName)
        fw2.write(convertResult)


    fw.close()

def dataUs(infos):
    return infos.split(".")[0]

def convert_annotation(path,classes,imagePath,deputyFileName): 
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
            result += " %s,%s,%s,%s,%d"%(xmin,ymin,xmax,ymax,classNum)
    if(hasClass):
        FileName = os.path.basename(path)
        FileName = os.path.splitext(FileName)[0]
        FileName = FileName+"."+deputyFileName
        result = "%s%s\n"%((imagePath + FileName),result)
    return result

if __name__ == "__main__":
    _main()