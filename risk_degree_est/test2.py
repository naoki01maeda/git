
import numpy as np
#import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

from PIL import Image
#from glob import glob
import glob
import xml.etree.ElementTree as ET 
import torch
import torchvision
from torchvision import transforms
import datetime
import yaml

from model import model
from test_dataloader import test_dataloader

def iou(score, box):
    under_score_list = []
    remove_list = []
    remain_list = []
    for i in range(len(box)):
        ax_mn, ay_mn, ax_mx, ay_mx = box[i][0], box[i][1], box[i][2], box[i][3]
        for n in range(len(box)):
            
            if n == i:
                continue
            
            bx_mn, by_mn, bx_mx, by_mx = box[n][0], box[n][1], box[n][2], box[n][3]

            a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
            b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

            abx_mn = max(ax_mn, bx_mn)
            aby_mn = max(ay_mn, by_mn)
            abx_mx = min(ax_mx, bx_mx)
            aby_mx = min(ay_mx, by_mx)
            w = max(0, abx_mx - abx_mn + 1)
            h = max(0, aby_mx - aby_mn + 1)
            intersect = w*h

            iou = intersect / (a_area + b_area - intersect)
            if iou >= 0.4:
                if score[i] < score[n]:
                    remove_list.append(i)
                    break
    for r in range(len(box)):
        if r not in remove_list:
            remain_list.append(r)
            
    #print(remain_list)
    return remain_list

def test2(train_model,batch_size,image_path,output_dir):
    anno_region = "./data/ori_data/region_descriptions.json"#画像データ
    anno_img = "./data/ori_data/image_data.json"#領域データ
    img = "./data/ori_data/drama_image"#画像ファイルの場所("C:/Users/naoki/anaconda_work/Faster-R-CNN-pytorch-main/data/processing_image")(./data/ori_data/drama_image)
    mv = "./data/ori_data/drama_clip"#動画ファイルの場所

    data_ALL=[[anno_region,anno_img,img,mv]]
    
    day = datetime.datetime.now()

    folder=f'output_{day.day}_{day.hour}_{day.minute}'
        
    dataloder=test_dataloader(data_ALL,batch_size)
    
    model=torch.load(train_model)
    

    with open('class_label.json', 'r') as json_file:
        data_classes_list = json.load(json_file)
    dataset_class = data_classes_list

    data_class=dataset_class
    data_class.insert(0, "__background__")
    classes = tuple(data_class)
    colors = ((0,0,0),(255,0,0),(0,255,0),(0,0,255),(100,100,100),(50,50,50),(255,255,0),(255,0,255),(0,255,255),(100,100,0),(0,100,100))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    model.to(device)

    model.eval()

    #bbox表示の閾値
    s=0.1

    ALL_box=0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for vision_data,image_ids in tqdm(dataloder):#バッチごとに処理
        img_list = [img_data["image"].to(device) for img_data in vision_data]
        clip_list = [clip_data["clip"].to(device) for clip_data in vision_data]
        
        vision_batch = {"image":img_list, "clip":clip_list}
        #print(vision_batch["image"])
        #images = list(image.to(device) for image in images)
        with torch.no_grad():
            prediction = model(vision_batch)
            #print(len(prediction))#batchsizeと同じ
            #print(prediction[0].keys())#dict_keys(['boxes', 'labels', 'scores'])
            #print(prediction[0]['boxes'].shape)#torch.Size([検出数, 4])
            #print(prediction[0]['scores'].shape)#torch.Size([検出数])
        for j in range(len(vision_batch["image"])):#1バッチ内の1枚ごとに処理
            
            imgfile=image_path+'/'+str(image_ids[j])+'.jpg'
            print(imgfile)
            img = cv2.imread(imgfile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            num_boxs=0
            
            remove_list = iou(prediction[j]['scores'], prediction[j]['boxes'])
            #pos = torch.where(
            filter_box_pred = prediction[j]['boxes'][remove_list]
            filter_score_pred = prediction[j]['scores'][remove_list]
            for i,box in enumerate(filter_box_pred):
                

                score = filter_score_pred[i].cpu().numpy()
                if score > s:
                    #print("1")
                    score = round(float(score),2)
                    #cat = prediction[j]['labels'][i].cpu().numpy()
                    txt = '{}'.format(str(score))#txt = '{} {}'.format(classes[int(cat)], str(score))
                    #print(txt)
                    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                    #c = colors[float(score)]
                    box=box.cpu().numpy().astype('int')
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (int(score * 255),255-int(score * 255),0) , 2)
                    cv2.rectangle(img,(box[0], box[1] - cat_size[1] - 2),(box[0] + cat_size[0], box[1] - 2), (int(score * 255),255-int(score * 255),0), -1)
                    cv2.putText(img, txt, (box[0], box[1] - 2), font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    num_boxs+=1
                    ALL_box+=1

            #plt.figure(figsize=(15,10))
            #plt.imshow(img)
            #plt.show()
            #exit()
            t=f'train_model:{train_model} num_box:{num_boxs}'
            cat_size = cv2.getTextSize(t, font, 0.5, 2)[0]
            cv2.rectangle(img,(15, 15 - cat_size[1] - 2),(15+ cat_size[0], 15 +2), (255,255,255), -1)
            cv2.putText(img,t,(15,15),font, 0.5, (0, 0,0), thickness=1, lineType=cv2.LINE_AA)
            if not os.path.exists(f'./output/{folder}'):  # ディレクトリが存在しない場合、作成する。
                os.makedirs(f'./output/{folder}')
            #BGR RGB変換して保存
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./output/{folder}/{image_ids[j]}.jpg",img)

            #モデル情報の出力
            with open(f'./output/{folder}/model.txt', 'w') as f:
                print(f'test_data:{data_ALL}',file=f)
                print(f'batch_size:{batch_size}',file=f)
                print(f'train_model:{train_model}',file=f)
                print(f'Threshold:{s}',file=f)
                print(f'ALL_box:{ALL_box}',file=f)

        #exit()

    #plt.show()

    #gif画像化
    from makegif import make_gif
    print("gif作成中....")
    make_gif(f'./output/{folder}')

    

#test('4class','./frcnn_4class.pt',path,folder)