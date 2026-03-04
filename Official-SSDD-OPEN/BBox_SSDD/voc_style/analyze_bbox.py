# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import matplotlib
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
bbox_xmin_list = []
bbox_xmax_list = []
bbox_ymin_list = []
bbox_ymax_list = []

for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + '/' + image_pre + '.jpg'
    xmlfile = AnnoPath + '/'  + image_pre + '.xml'
    
    im = cv2.imread(imgfile)
    
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    
    filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
    
    size = annotation.getElementsByTagName('size')[0]
    
    # print('size =', size)
    
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    width_list.append(width)
    height_list.append(height)
    
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
            
            bbox_xmin_list.append(xmin)
            bbox_xmax_list.append(xmax)
            bbox_ymin_list.append(ymin)
            bbox_ymax_list.append(ymax)
    
    # print('width, height = ', width, height)

w_np_list = np.array(bbox_xmax_list) - np.array(bbox_xmin_list)
h_np_list = np.array(bbox_ymax_list) - np.array(bbox_ymin_list)

area_np_list = w_np_list * h_np_list

print('w_np_list = ', w_np_list)

center_x_np_list = (np.array(bbox_xmax_list) + np.array(bbox_xmin_list)) // 2
center_y_np_list = (np.array(bbox_ymax_list) + np.array(bbox_ymin_list)) // 2

max_w = int(np.max(w_np_list))
max_h = int(np.max(h_np_list))

min_w = int(np.min(w_np_list))
min_h = int(np.min(h_np_list))

mean_w = int(np.mean(w_np_list))
mean_h = int(np.mean(h_np_list))

median_w = int(np.median(w_np_list))
median_h = int(np.median(h_np_list))

# mode_w = int(np.argmax(np.bincount(w_np_list)))
# mode_h = int(np.argmax(np.bincount(h_np_list)))

plt.figure(1)
s1 = plt.scatter(w_np_list, h_np_list, c='darkorchid', marker='o')
s2 = plt.scatter(mean_w, mean_h, s=40, c='r', marker='p')
s3 = plt.scatter(min_w, min_h, s=40, c='g', marker='p')
s4 = plt.scatter(max_w, max_h, s=40, c='b', marker='p')
s5 = plt.scatter(median_w, median_h, s=40, c='k', marker='p')
# s6 = plt.scatter(mode_w, mode_h, s=40, c='m', marker='p')

lines_x = [0, 400]
lines_y = [0, 400]

plt.plot(lines_x, lines_y, c='g')


plt.legend((s1, s2, s3, s4, s5), \
('BBox Ship, Width, Height', \
'BBox Ship ,Width, Height | mean    = {}, {}'.format(mean_w, mean_h),
'BBox Ship, Width, Height | min       = {}, {}'.format(min_w, min_h),
'BBox Ship, Width, Height | max      = {}, {}'.format(max_w, max_h),
'BBox Ship, Width, Height | median = {}, {}'.format(median_w, median_h),
# 'w, h | mode    = {}, {}'.format(mode_w, mode_h),
), \
loc='best')

plt.xlim(0, 400)
plt.ylim(0, 400)

plt.xlabel('BBox Ship Width', font)
plt.ylabel('BBox Ship Height', font)
plt.savefig('BBox_Ship_Width_Height.png', bbox_inches='tight')
plt.show()

ratio_w_h_list = w_np_list / h_np_list

# print(ratio_w_h_list)


max_ratio_width_height = np.max(ratio_w_h_list)
min_ratio_width_height = np.min(ratio_w_h_list)
mean_ratio_width_height = np.mean(ratio_w_h_list)

# print(ratio_w_h_list, ratio_w_h_list)
# ------------------------------------------------------------------------------------
plt.figure(2)

arr=plt.hist(ratio_w_h_list, bins=20, color='teal', edgecolor='w')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])))
    
plt.xlim(0.1, 7.5)
plt.ylim(0, 1100)

plt.xlabel('Ratio between BBox Ship Width and Height', font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_BBox_Ship_Width_Height.png', bbox_inches='tight')
plt.show()



# ------------------------------------------------------------------------------------
plt.figure(3)
area_np_list = area_np_list / 10000
arr=plt.hist(area_np_list, bins=50, color='brown', edgecolor='w')
# print('arr = ', arr)

for i in range(50):
    
    
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])), fontsize=4.5)
    
plt.xlim(0, (55000/10000))
plt.ylim(0, 1700)

# arr.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.xlabel('Area of BBox Ship ' + "(×${10}^{4}$)", font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_Area_BBox_Ship.png', bbox_inches='tight', dpi=1200)
plt.show()

# ------------------------------------------------------------------------------------
plt.figure(4)
s1 = plt.scatter(center_x_np_list, center_y_np_list, c='darkslategray', marker='o')

max_cx = int(np.max(center_x_np_list))
max_cy = int(np.max(center_y_np_list))

min_cx = int(np.min(center_x_np_list))
min_cy = int(np.min(center_y_np_list))

mean_cx = int(np.mean(center_x_np_list))
mean_cy = int(np.mean(center_y_np_list))

median_cx = int(np.median(center_x_np_list))
median_cy = int(np.median(center_y_np_list))

s2 = plt.scatter(mean_cx, mean_cy, s=40, c='r', marker='p')
s3 = plt.scatter(min_cx, min_cy, s=40, c='g', marker='p')
s4 = plt.scatter(max_cx, max_cy, s=40, c='b', marker='p')
s5 = plt.scatter(median_cx, median_cy, s=40, c='k', marker='p')
# s6 = plt.scatter(mode_w, mode_h, s=40, c='m', marker='p')

lines_x = [0, 550]
lines_y = [0, 550]

plt.plot(lines_x, lines_y, c='g')


plt.legend((s1, s2, s3, s4, s5), \
('BBox Ship, Center x, Center y,', \
'BBox Ship, Center x, Center y| mean    = {}, {}'.format(mean_cx, mean_cy),
'BBox Ship, Center x, Center y | min       = {}, {}'.format(min_cx, min_cy),
'BBox Ship, Center x, Center y | max      = {}, {}'.format(max_cx, max_cy),
'BBox Ship, Center x, Center y | median = {}, {}'.format(median_cx, median_cy),
# 'w, h | mode    = {}, {}'.format(mode_w, mode_h),
), \
loc='best')

plt.xlim(0, 550)
plt.ylim(0, 550)

plt.xlabel('BBox Ship Center x', font)
plt.ylabel('BBox Ship Center y', font)
plt.savefig('BBox_Ship_Center_x_Center_y.png', bbox_inches='tight')
plt.show()





