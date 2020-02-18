import xml.etree.ElementTree as ET
import os
from os import getcwd
import sys

def _main():
    path = sys.argv[1] #"./Data/Annotations/"
    imagePath = sys.argv[2]#"./Data/JPEGImages/"
    writePath = sys.argv[3]+"train.txt"#"./model_data/train.txt"
    deputyFileName = "jpg"
    # classes = ["bicycle","car","cat","dog","person"]
    fr = open(sys.argv[4] , 'r')#"model_data/voc_classes.txt"
    classes = fr.read().split("\n")
    fr.close()

    fw = open(writePath, "w")
    for fileName in os.listdir(path):
        if fileName in ".gitignore":
            continue
        print("readFile:",(path+fileName))
        convertResult = convert_annotation((path+fileName),classes,imagePath,deputyFileName)
        fw.write(convertResult)
    fw.close()

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
                xmin = xmlObj2.find('xmin').text.replace(" ", "").replace("\t", "").replace("\n", "").replace(" ", "").replace("\t", "").replace("\n", "")
                ymin = xmlObj2.find('ymin').text.replace(" ", "").replace("\t", "").replace("\n", "")
                xmax = xmlObj2.find('xmax').text.replace(" ", "").replace("\t", "").replace("\n", "")
                ymax = xmlObj2.find('ymax').text.replace(" ", "").replace("\t", "").replace("\n", "")
            result += " %s,%s,%s,%s,%d"%(xmin,ymin,xmax,ymax,classNum)
    if(hasClass):
        FileName = os.path.basename(path)
        FileName = os.path.splitext(FileName)[0]
        FileName = FileName+"."+deputyFileName
        result = "%s%s\n"%((imagePath + FileName),result)
    return result

if __name__ == "__main__":
    _main()