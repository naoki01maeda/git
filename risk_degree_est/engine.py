import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

#from coco_utils import get_coco_api_from_dataset
#from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()#学習モードにする
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    loss_epo=[]

    loss_class=[]
    loss_box_reg=[]
    loss_obje=[]
    loss_rpn_reg=[]


    for vision_data, targets,image_ids in metric_logger.log_every(data_loader, print_freq, header):
        
        img_list = [img_data["image"].to(device) for img_data in vision_data]
        clip_list = [clip_data["clip"].to(device) for clip_data in vision_data]
        
        vision_batch = {"image":img_list, "clip":clip_list}#入力データ
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]#正解データ
        
        try:
            loss_dict = model(vision_batch, targets)
            #print(loss_dict)#{'loss_classifier'(fastRCNN): tensor(0.2389, device='cuda:0', grad_fn=<MeanBackward0>), 'loss_box_reg'(fastRCNN): tensor(0.0080, device='cuda:0', grad_fn=<DivBackward0>), 
            # 'loss_objectness'(rpn): tensor(0.7055, device='cuda:0',grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg'(rpn): tensor(0.0062, device='cuda:0', grad_fn=<DivBackward0>)}
        #boxが0個だった時のデバッグ
        except ValueError:
            #print(image_ids)
            pass
        else:
            
            losses = sum(loss for loss in loss_dict.values())
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            loss_epo.append(loss_value)

            #各lossの出力
            loss_value_c=loss_dict_reduced['loss_classifier'].item()#roi_heads
            loss_value_b=loss_dict_reduced['loss_box_reg'].item()#roi_heads
            loss_value_o=loss_dict_reduced['loss_objectness'].item()#rpn
            loss_value_r=loss_dict_reduced['loss_rpn_box_reg'].item()#rpn


            loss_class.append(loss_value_c)
            loss_box_reg.append(loss_value_b)
            loss_obje.append(loss_value_o)
            loss_rpn_reg.append(loss_value_r)







            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        

    return loss_epo, loss_class



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    #print(data_loader.dataset[0])
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)


    for image, targets ,image_ids in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
