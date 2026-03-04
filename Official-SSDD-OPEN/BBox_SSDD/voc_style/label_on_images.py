# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
  
def read_xml(ImgPath = 'JPEGImages_test/', AnnoPath = 'Annotations_test/', Savepath = 'JPEGImages_test_BBox_GT/'):

    if not os.path.isdir(Savepath):
        os.makedirs(Savepath)

    imagelist = os.listdir(AnnoPath)

    for image in imagelist:  
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + '/' + image_pre + '.jpg'
        xmlfile = AnnoPath + '/'  + image_pre + '.xml'
        im = cv2.imread(imgfile)
        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            bndbox = objects.getElementsByTagName('bndbox')
            for box in bndbox:
                    xmin_list = box.getElementsByTagName('xmin')
                    xmin = int(xmin_list[0].childNodes[0].data)
                    ymin_list = box.getElementsByTagName('ymin')
                    ymin = int(ymin_list[0].childNodes[0].data)
                    xmax_list = box.getElementsByTagName('xmax')
                    xmax = int(xmax_list[0].childNodes[0].data)
                    ymax_list = box.getElementsByTagName('ymax')
                    ymax = int(ymax_list[0].childNodes[0].data)

                    cv2.rectangle(im,(xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
                    # cv2.putText(im, 'ship', (xmin,ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
        path = Savepath + '/' + image_pre + '.jpg'
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(im, objectname, (xmin,ymin - 7), font, 0.5, (0, 0, 255), 1)
                    
        cv2.imwrite(path, im)
read_xml()