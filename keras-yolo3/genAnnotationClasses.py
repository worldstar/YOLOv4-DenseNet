import xml.etree.ElementTree as ET
import os
from os import getcwd

def _main():
    path = "./Data/Annotations/"
    writePath = "./model_data/voc_classes.txt"

    Classes = getAnnotationClasses(path)

    fw = open(writePath, "w")
    for i in range(0,len(Classes),1):
        if(i == len(Classes)-1):
            fw.write(Classes[i])
        else:
            fw.write(Classes[i] + "\n")
    fw.close()

def getAnnotationClasses(path):
    result = []
    for fileName in os.listdir(path):
        print('readFile:',(path+fileName))
        try: 
            xmlFile = open((path+fileName)) 
        except:
            xmlFile = open((path+fileName),encoding="utf-8")         
        xmlTree = ET.parse(xmlFile)
        xmlRoot = xmlTree.getroot()
        for xmlObj in xmlRoot.iter('object'):
            name = xmlObj.find('name').text
            isClass = False
            for i in range(0,len(result),1):
                if(name == result[i]):
                    isClass = True
            if(isClass == False):
                result.append(name)
    return result

if __name__ == "__main__":
    _main()