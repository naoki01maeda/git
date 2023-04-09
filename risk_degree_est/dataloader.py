'''
さらに高速化
背景のみの時の改善
スケール、バッジサイズを引数指定可能

'''
import numpy as np
import re
from natsort import natsorted
#import pandas as pd
 
from PIL import Image
#from glob import glob
import glob
import xml.etree.ElementTree as ET 
import cv2
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import TensorDataset
import os
import time
import yaml
import json

class xml2list(object):
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        xml = ET.parse(xml_path).getroot()
        
        #たぶんいらない
        #for size in xml.iter("size"):     
        #    width = float(size.find("width").text)
        #    height = float(size.find("height").text)
    
        ###################################################################################################
        boxes = []
        labels = []
        zz=0
        
        for zz,obj in enumerate(xml.iter('object')):
            
            label = obj.find('name').text
           
            #print(label)
            ##指定クラスのみ
            #classes2= ['car', 'person', 'bike','motor','rider','truck', 'bus'] ##bdd100kをhokkaido対応させる応急処置  
            classes2=[] 
            if label in self.classes :
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.classes.index(label))
            elif label in classes2:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                if label=='rider':
                    #riderはperson判定
                    labels.append(classes2.index('person'))
                elif label=='truck' or label=='bus':
                    #truck,busはcar判定
                    labels.append(classes2.index('car')) 
                else:                   
                    labels.append(classes2.index(label))
            else:
                continue
            
        num_objs = zz +1

        ##BBOXが０の時がある、、
        #annotations = {'image':img, 'bboxes':boxes, 'labels':labels}
        anno = {'bboxes':boxes, 'labels':labels}

        return anno,num_objs
        ##################################################################################################




class MyDataset(torch.utils.data.Dataset):
        
    
        #def __init__(self, df, image_dir):
        def __init__(self,image_dir1, mv_dir1, image_paths,region_paths,scale,classes):
            
            super().__init__()
            self.image_ids = []
            self.image_dir = image_dir1
            self.mv_dir = mv_dir1#"./data/ori_data/drama_clip"
            self.image_paths = image_paths
            self.region_paths = region_paths
            self.region_data = json.load(open(self.region_paths, 'r'))
            self.image_json_data = json.load(open(self.image_paths, 'r'))
            for i in self.image_json_data:
                if len([path for path in natsorted(glob.glob("{}/{}/*.jpg".format(self.mv_dir, i["image_id"]-1)))]) >= 6:
                    self.image_ids.append(i["image_id"])
            self.image_ids = self.image_ids[20:]
            #print(self.image_ids)
            self.scale=scale
            self.classes=classes
            
            self.mv_use = True
        
        def resize_img(self, img, scale, transform):
            #print(img.size)#(1000, 740)
            ##################画像のスケール変換######################
            t_scale_tate=scale ##目標のスケール(縦)
            #縮小比を計算
            ratio=t_scale_tate/img.size[1]
            ##目標横スケールを計算
            t_scale_yoko=img.size[0]*ratio
            t_scale_yoko=int(t_scale_yoko)
            
            #print('縮小前:',image.size)
            #print('縮小率:',ratio)
            #リサイズ
            img = img.resize((t_scale_yoko,t_scale_tate))
            #print('縮小後:',img.size)
            #########################################################
            img = transform(img)
            #print(img.shape)
            return img, ratio

        def __getitem__(self, index):
    
            transform = transforms.Compose([
                                            transforms.ToTensor()
            ])
    
            # 入力画像の読み込み
            ###############################################################
            
            image_id=self.image_ids[index]
            #print(image_id,"image_id")
            image = Image.open(f"{self.image_dir}/{image_id}.jpg")
            
            image, ratio = self.resize_img(image, self.scale, transform)
            
            ###############################################################
            
            #動画の読み込み
            ###############################################################
            clip = []
            if self.mv_use == True:
                mv_id = self.image_ids[index]
                #print(mv_id-1,"clip")
                files = [path for path in natsorted(glob.glob("{}/{}/*.jpg".format(self.mv_dir, mv_id-1)))]
                files_filter = files[-6:]
                
                for p in files_filter:
                    one_frame = Image.open(p)
                    frame, _ = self.resize_img(one_frame, self.scale, transform)
                    clip.append(frame)
                clip = torch.stack(clip, dim = 1)
            else:
                clip = torch.tensor(clip)
            ###############################################################
            
            #box、ラベル読み込み
            ###############################################################
            boxes_data = []
            labels_data = []
            region_counter = 0
            
            for one_data in self.region_data:
                if one_data["id"] == image_id:
                    #print(one_data["id"], "region")
                    for one_region_data in one_data["regions"]:
                        boxes_data.append(one_region_data["boxes"])
                        labels_data.append(float(one_region_data["risk"]))
                        region_counter += 1
                    break
           
            boxes = torch.as_tensor(boxes_data, dtype=torch.int64)
            labels = torch.as_tensor(labels_data, dtype=torch.float64)
            ###############################################################
            #print(labels)
            #no-bbox
            if len(boxes)==0:
                
                iscrowd = torch.zeros((region_counter,), dtype=torch.int64)
                #area=[0]
                area = torch.as_tensor([[0]], dtype=torch.float32)
                vision_data = {}
                vision_data["image"] = image
                vision_data["clip"] = clip
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["image_id"] = torch.tensor(self.image_ids[index])
                target["area"] = area
                target["iscrowd"] = iscrowd
                #print(image_id)
                return vision_data, target, image_id

            else:
                
                #bboxの縮小
                #print('縮小前:',boxes)
                boxes=boxes*ratio
                #print('縮小後:',boxes)
                #print(ratio)
                area = (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2]-boxes[:, 0])
                area = torch.as_tensor(area, dtype=torch.float32)

                iscrowd = torch.zeros((region_counter,), dtype=torch.int64)

                #print(labels,"111")
                #print(image_id)
                #print(area)
                #print(iscrowd)
                vision_data = {}
                vision_data["image"] = image
                vision_data["clip"] = clip
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["image_id"] = torch.tensor(self.image_ids[index])
                target["area"] = area
                target["iscrowd"] = iscrowd
                #print(target,"============")
                
                return vision_data, target, image_id
        
        def __len__(self):
            #return self.image_ids.shape[0]
            return len(self.image_ids)
        
def dataloader (data,batch_size,scale=720,shuffle=True):
    itr=0
    #print(len(data))
    for d in data:
        #print(len(d))
        region_paths=d[0]
        image_paths=d[1]
        image_dir1=d[2]
        mv_dir1 = d[3]

        with open('class_label.json', 'r') as json_file:
            data_classes_list = json.load(json_file)

        classes = data_classes_list
        #print(classes)
        """
        if dataset_class =='bdd100k':
            classes = ['person', 'traffic light', 'train', 'traffic sign', 'rider', 'car', 'bike', 'motor', 'truck', 'bus']
        elif dataset_class == 'hokkaido' or  dataset_class == 'original_VOC':
            classes = ['car', 'bus', 'person', 'bicycle', 'motorbike', 'train']
        elif dataset_class == '4class':
            classes = ['car', 'person', 'bicycle', 'motorbike']
        else:
            print('Error : please choose dataset bdd100K or hokkaido')
            exit()
        """

        #dataset = MyDataset(df, image_dir1)
        dataset = MyDataset(image_dir1, mv_dir1, image_paths, region_paths, scale, classes)
        
        

        

        #データのロード
        torch.manual_seed(2020)
        
        if itr == 0:
            train=dataset
        else:
            train=torch.utils.data.ConcatDataset([train,dataset])
        itr=itr+1
        
        #訓練と検証で分ける
        train_size = int(0.8 * len(train))
        val_size = len(train) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
                                    train, [train_size, val_size])
    

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn, pin_memory=True)#3
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn, pin_memory=True)
    #train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)#3

    return train_dataloader, val_dataloader

    #データセットテスト用 
    #return train_dataloader,dataset
    



#テスト

#データの場所



bdd_val_xml="C:/Research/dataset/BDD100K/annotation/val"
bdd_val_img="C:/Research/dataset/BDD100K/images/100k/val"

data_ALL=[[bdd_val_xml,bdd_val_img]]
dataset_class='bdd100k'

#data_ALL=[[original_VOC_xml,original_VOC_img],[bdd_xml,bdd_img]]

#テスト兼ラベルの抽出(batchsize=1)でないとだめ
'''
t=dataloader_v6(data_ALL,dataset_class,3)
for images, targets,image_ids in t:
    
    
    for t in targets:
        targets = {k: v for k, v in t.items()}
        if 3 in targets['labels'] :
            print(image_ids[0])
            #exit()
'''
  




'''
###
#表示実験したい場合はdataloader_v6でdatasetを出力
t,dataset=dataloader_v6(data_ALL,dataset_class,3)
import matplotlib.pyplot as plt
import cv2
classes = ('__background__','car', 'person', 'bicycle', 'motorbike')
colors = ((0,0,0),(255,0,0),(0,255,0),(0,0,255),(100,100,100))
#image1, target,image_id = dataset[470]###null
image1, target,image_id = dataset[0]
#tensor2numpy
image1 = image1.to('cpu').detach().numpy().copy()
#print('------------------------')
image1=image1*255
image1=image1.transpose(1, 2, 0)
image1 = np.ascontiguousarray(image1, dtype=np.uint8)


#print(image1)
#image1 = image1.mul(255).permute(1, 2, 0).byte().numpy()
labels = target['labels'].cpu().numpy()
boxes = target['boxes'].cpu().numpy()
boxes=boxes.astype(np.int64)


for i,box in enumerate(boxes):   
    txt = classes[labels[i]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c = colors[labels[i]]
    cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), c , 2)
    cv2.rectangle(image1,(box[0], box[1] - cat_size[1] - 2),(box[0] + cat_size[0], box[1] - 2), c, -1)
    cv2.putText(image1, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)




plt.figure(figsize=(20,20))

plt.imshow(image1)

plt.show()

'''