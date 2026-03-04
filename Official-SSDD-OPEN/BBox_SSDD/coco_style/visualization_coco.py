import os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

json_file = 'annotations/train.json'
dataset_dir = 'images/train/'

coco = COCO(json_file)
catIds = coco.getCatIds(catNms=['ship']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值

for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread(dataset_dir + img['file_name'])
    
    plt.axis('off')
    plt.imshow(I) #绘制图像，显示交给plt.show()处理
    
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    
    plt.show() #显示图像