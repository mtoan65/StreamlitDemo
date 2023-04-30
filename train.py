#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from utils.set_seeds import seed_everything
from utils.read_dataset import read_dataset
from utils.train_model import train
from config import seed, batch_size, root, model_path, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
import argparse
from pytorch_metric_learning import losses, miners
from models.mobilenet import mobilenetv2, mobilenetv3_small, mobilenetv3_large
from models.resnet import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from models.Resnet18 import resnet18
from models.Resnet34 import resnet34
# from models.sg_prj_resnet import sg_prj_resnet50, sg_prj_resnet101, sg_prj_resnet152, sg_prj_resnext50_32x4d, sg_prj_resnext101_32x8d, sg_prj_resnext101_64x4d,\
#     sg_prj_wide_resnet50_2, sg_prj_wide_resnet101_2
# from models.mt_prj_resnet import mt_prj_resnet50, mt_prj_resnet101, mt_prj_resnet152, mt_prj_resnext50_32x4d, mt_prj_resnext101_32x8d, mt_prj_resnext101_64x4d,\
#     mt_prj_wide_resnet50_2, mt_prj_wide_resnet101_2


device = torch.device("cuda")
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="chosen model")
args = vars(ap.parse_args())


model_pool = {
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext101_64x4d': resnext101_64x4d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'mobilenetv2': mobilenetv2,
    'mobilenetv3_small':mobilenetv3_small,
    'mobilenetv3_large':mobilenetv3_large
}

pretrained_url_pool = dict.fromkeys(['resnet18'], "https://download.pytorch.org/models/resnet18-f37072fd.pth")
pretrained_url_pool.update(dict.fromkeys(['resnet34'], "https://download.pytorch.org/models/resnet34-b627a593.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnet50'], "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnet101'], "https://download.pytorch.org/models/resnet101-cd907fc2.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnet152'], "https://download.pytorch.org/models/resnet152-f82ba261.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext50_32x4d'], "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext101_32x8d'], "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext101_64x4d'], "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth"))
pretrained_url_pool.update(dict.fromkeys(['wide_resnet50_2'], "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"))
pretrained_url_pool.update(dict.fromkeys(['wide_resnet101_2'], "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth"))
pretrained_url_pool.update(dict.fromkeys(['mobilenetv2'], "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"))
pretrained_url_pool.update(dict.fromkeys(['mobilenetv3_small'], "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"))
pretrained_url_pool.update(dict.fromkeys(['mobilenetv3_large'], "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"))


def main():
    # set all the necessary seeds
    seed_everything(seed)
    # Read the dataset
    # trainloader, valloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)
    trainloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)

    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), pretrained=True)
    try: 
      have_prj = model.have_prj
    except:
      have_prj = None
    # define the CE loss function
    criterion = nn.CrossEntropyLoss()

    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    parameters = model.parameters()

    # define the optimizer
    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=0.9, weight_decay=weight_decay)
    # define the learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate, verbose=True)

    # loading checkpoint
    save_path = os.path.join(model_path, args["model"])
    if os.path.exists(save_path):
        start_epoch, best_val_acc = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        best_val_acc = 0.0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))
     # Train the model
    train(model=model,
          device=device,
          have_prj=have_prj,
          trainloader=trainloader,
        #   valloader=valloader,
          testloader=testloader,
          metric_loss=metric_loss,
          miner=miner,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          best_val_acc = best_val_acc)


if __name__ == '__main__':
    main()