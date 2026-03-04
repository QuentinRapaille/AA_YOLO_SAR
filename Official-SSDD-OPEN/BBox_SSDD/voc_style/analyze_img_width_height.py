# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

font = {'family' : 'Palatino Linotype',
'weight' : 'normal',
'size'   : 15,
}








ImgPath = 'JPEGImages_test/'
AnnoPath = 'Annotations_test/'

imagelist = os.listdir(AnnoPath)
width_list = []
height_list = []

for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + '/' + image_pre + '.jpg'
    xmlfile = AnnoPath + '/'  + image_pre + '.xml'
    
    im = cv2.imread(imgfile)
    
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    
    filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
    
    size = annotation.getElementsByTagName('size')[0]
    
    
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    width_list.append(width)
    height_list.append(height)
    
    # print('width, height = ', width, height)

max_width = int(np.max(width_list))
max_height = int(np.max(height_list))

min_width = int(np.min(width_list))
min_height = int(np.min(height_list))

mean_width = int(np.mean(width_list))
mean_height = int(np.mean(height_list))

median_width = int(np.median(width_list))
median_height = int(np.median(height_list))

mode_width = int(np.argmax(np.bincount(width_list)))
mode_height = int(np.argmax(np.bincount(height_list)))

plt.figure(1)
s1 = plt.scatter(width_list, height_list, c='peru', marker='o')
s2 = plt.scatter(mean_width, mean_height, s=40, c='r', marker='p')
s3 = plt.scatter(min_width, min_height, s=40, c='g', marker='p')
s4 = plt.scatter(max_width, max_height, s=40, c='b', marker='p')
s5 = plt.scatter(median_width, median_height, s=40, c='k', marker='p')
s6 = plt.scatter(mode_width, mode_height, s=40, c='m', marker='p')

lines_x = [150, 550]
lines_y = [150, 550]

plt.plot(lines_x, lines_y, c='g')


plt.legend((s1, s2, s3, s4, s5, s6), \
('Image Width, Height', \
'Image Width, Height | mean    = {}, {}'.format(mean_width, mean_height),
'Image Width, Height | min       = {}, {}'.format(min_width, min_height),
'Image Width, Height | max      = {}, {}'.format(max_width, max_height),
'Image Width, Height | median = {}, {}'.format(median_width, median_height),
'Image Width, Height | mode    = {}, {}'.format(mode_width, mode_height),
), \
loc='best')

plt.xlim(200, 700)
plt.ylim(150, 550)

plt.xlabel('Image Width', font)
plt.ylabel('Image Height', font)
plt.savefig('Image_Width_Height.png', bbox_inches='tight')
plt.show()

ratio_width_height_list = np.array(width_list) / np.array(height_list)
print(ratio_width_height_list)


max_ratio_width_height = np.max(ratio_width_height_list)
min_ratio_width_height = np.min(ratio_width_height_list)
mean_ratio_width_height = np.mean(ratio_width_height_list)

print(max_ratio_width_height, min_ratio_width_height)

plt.figure(2)
# bins = [0.50, 0.75, 1.00, 1.25, 1.50, 1.750, 2.00, 2.25, 2.50, 2.750, 3.00]

arr=plt.hist(ratio_width_height_list, bins=20, color='deepskyblue', edgecolor='w')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])))
    
plt.xlim(0.75, 2.6)
plt.ylim(0, 300)

plt.xlabel('Ratio between Image Width and Image Height', font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_Image_Width_Height.png', bbox_inches='tight')
plt.show()






