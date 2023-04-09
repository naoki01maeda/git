import numpy as np
import json
#import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 

import torch
import torchvision
#import vision.torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from resnet_backbone import ResNetBackBone

import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def model (model_name,backbone_model):
    #モデルの定義


    import torchvision
    from models.faster_rcnn import FasterRCNN
    from models.rpn import AnchorGenerator
    
    
    if backbone_model == "mobilenet":
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    elif backbone_model ==  "resnet":
        
        backbone=resnet_fpn_backbone("resnet50",weights='ResNet50_Weights.DEFAULT')#IMAGENET1K_V1で事前学習されている(https://qiita.com/noobar/items/6d985f92ad6e364aaefe参照)
        
        backbone.out_channels = 256
    else :
        print('Error: please choose backbone mobilenet or resnet')
        exit()
    

    
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))#アンカーボックスのサイズの種類
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)#アンカーボックスのアスペクト比の種類 * アンカーボックスのサイズの種類
            #合計のアンカーボックスは、anchor_sizes　* aspect_ratios = 5 * 3 = 15個
    anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios)#参照(https://tanalib.com/faster-rcnn-anchor/)

        
    # put the pieces together inside a FasterRCNN model

    
    with open('class_label.json', 'r') as json_file:
        data_classes_list = json.load(json_file)
    
    classes = data_classes_list
    
    num_classes=len(classes)+1
    

    if model_name=='Risk_estimation_Network':
        
    
        model = FasterRCNN(backbone,#特徴抽出機となるバックボーン
                        num_classes=num_classes,#分類クラス数
                        rpn_anchor_generator=anchor_generator)
        
    
    else:
        print('Error: Choose the model name ')
        exit()

  
    return model

