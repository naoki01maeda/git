#https://github.com/Ryunosuke-Ikeda/Faster-R-CNN-pytorch
from trainer import trainer
from evaluator import evaluator
from evaluator import eval_loss_plot
from test2 import test2
import argparse
from pathlib import Path
import torch
import yaml
import os
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#from model import model

def get_args_parser():
    parser = argparse.ArgumentParser('Set frcnn detector', add_help=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=101, type=int)
    parser.add_argument('--batchsize', default=3, type=int)
    parser.add_argument('--backbone',default='resnet')

    parser.add_argument('--model', default='Risk_estimation_Network',
                        help='Choose the models')
    parser.add_argument('--dataset_name', default='demo',
                        help='Choose the train dataset ')

    parser.add_argument('--val_dataset_name', default='demo',
                        help='Choose the val dataset ')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--train_model_path', default='',
                        help='input the train model path')     
    parser.add_argument('--img_path', default='',
                        help='input the test image path')                  

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval_loss',action='store_true')

    
    
    return parser


def main(args):

    #引数受け取り
    epochs=args.epochs
    lr=args.lr
    batchsize=args.batchsize
    output_dir=args.output_dir
    model_name=args.model
    backbone=args.backbone
    
    if torch.cuda.is_available():
        print("GPUを使用できます")
    else:
        print("GPUは使用していません")
    
    anno_region = "./data/ori_data/region_descriptions.json"#画像データ
    anno_img = "./data/ori_data/image_data.json"#領域データ
    img = "./data/ori_data/drama_image"#画像ファイルの場所
    mv = "./data/ori_data/drama_clip"#動画ファイルの場所

    data_ALL=[[anno_region,anno_img,img,mv]]


    if args.eval:
        train_model=args.train_model_path
        evaluator(data_ALL,batchsize,train_model)
        print("============fin==============")
        return

    if args.test:
        train_model=args.train_model_path
        img_path=args.img_path
        test2(train_model,batchsize,img_path,output_dir)
        print("============fin==============")
        return

    if args.eval_loss:
        train_model=args.train_model_path
        eval_loss_plot(data_ALL,batchsize,train_model)
        print("============fin==============")
        return


    print("Start training")
    #modelの読み込み
    from model import model
    #学習済みデータ
    train_model=args.train_model_path

    if train_model=='':
        model=model(model_name,backbone)#モデル新規
    else:
        print(f'train_model:{train_model}')
        model=torch.load(train_model)#事前学習モデル読み込み
        
    #model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
    #model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)
    
    trainer(model,data_ALL,args)#モデル、データ、設定を渡す


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FRCNN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)