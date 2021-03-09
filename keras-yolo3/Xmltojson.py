### xml to csv
import cv2
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import json
import base64
import glob
import sys

xml_path         = sys.argv[1]#'logs/20200421_Y&D_Adam&1e-4_focalloss&gamma=2.^alpha=.25/'
# image_path       = sys.argv[2]
# write_path       = sys.argv[3]

def xml2csv(xml_path):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

    """
    # print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df=pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        # print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height','class','xmin','ymin','xmax','ymax'])
    return xml_df

def df2labelme(symbolDict,image_path,image):
    """ convert annotation in CSV format to labelme JSON

    Args:
        symbolDict (dataframe): annotations in dataframe
        image_path (str): path to image
        image (np.ndarray): image read as numpy array

    Returns:
        JSON: converted labelme JSON

    """
    try:
        symbolDict['min']= symbolDict[['xmin','ymin']].values.tolist()
        symbolDict['max']= symbolDict[['xmax','ymax']].values.tolist()
        symbolDict['points']= symbolDict[['min','max']].values.tolist()
        symbolDict['line_color'] = None
        symbolDict['fill_color'] = None
        symbolDict['shape_type']='rectangle'
        symbolDict['group_id']=None
        height,width,_=image.shape
        symbolDict['height']=height
        symbolDict['width']=width
        encoded = base64.b64encode(open(image_path, "rb").read())
        symbolDict.loc[:,'imageData'] = encoded
        symbolDict.rename(columns = {'class':'label','filename':'imagePath','height':'imageHeight','width':'imageWidth'},inplace=True)
        converted_json = (symbolDict.groupby(['imagePath','imageWidth','imageHeight','imageData'], as_index=False)
                     .apply(lambda x: x[['label','line_color','fill_color','points','shape_type','group_id']].to_dict('r'))
                     .reset_index()
                     .rename(columns={0:'shapes'})
                     .to_json(orient='records'))
        converted_json = json.loads(converted_json)[0]
        converted_json["lineColor"]=  [0,255,0,128]
        converted_json["fillColor"]=  [255,0,0,128]
    except Exception as e:
        converted_json={}
        print('error in labelme conversion:{}'.format(e))
    return converted_json

for file in glob.glob(xml_path):
    if file.lower().endswith(('.xml')):
        xml_csv   = xml2csv(file)
        jsonfile  = "m"+file.strip('.xml')+".json"
        imagefile = "m"+file.strip('.xml')+".png"
        image=cv2.imread(imagefile)
        csv_json=df2labelme(xml_csv,imagefile,image)
        with open(jsonfile, 'w') as outfile:
            json.dump(csv_json, outfile)

    # img = Image.open(jpgfile)
    # image = cv2.imread(jpgfile)
    # imagename = os.path.basename(jpgfile)
    # (h, w) = image.shape[:2]
    # create_tree(imagename, h, w)
    # root,pre,predictedarray = yolo.detect_imagexml(img,annotation)
    # if pre == True:
    #     for predicteditem in predictedarray:
    #         tree = ET.ElementTree(root)
    #         Path(SPath+"/"+predicteditem+"/").mkdir(parents=True, exist_ok=True)
    #         tree.write('.\{}\{}.xml'.format(SPath+predicteditem+"/", imagename.strip('.png')))
    #         img.save('.\{}\{}'.format(SPath+predicteditem+"/", imagename))
    # else:
    #     Path(SPath+"Normal/").mkdir(parents=True, exist_ok=True)
    #     img.save('.\{}\{}'.format(SPath+"Normal/", imagename))
    # yolo.detect_imagexml(img)

# xml_path='ZmRlwq1naGxm-11382573_1_1.mp4-2.xml'

# image_path='ZmRlwq1naGxm-11382573_1_1.mp4-2.png'
# image=cv2.imread(image_path)
# csv_json=df2labelme(xml_csv,image_path,image)
# with open('ZmRlwq1naGxm-11382573_1_1.mp4-2.json', 'w') as outfile:
#     json.dump(csv_json, outfile)

# print(csv_json)