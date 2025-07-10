import os
import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm

sys.path.append('./models')
import numpy as np
from datetime import datetime
# from models import CatNet
from model.MyNet import MyNet as MALNet
from torchvision.utils import make_grid
from data_cod import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./model/smt_tiny.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./Train/Imgs/', help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='./Train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='./Test/CAMO/Imgs/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./Test/CAMO/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ',opt.gpu_id)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

image_root = opt.rgb_root
gt_root = opt.gt_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root



save_path = opt.save_path

logging.basicConfig(filename=save_path + 'RGBD.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CATNet-Train_4_pairs")

# build the model
model = MALNet()

num_parms = 0
""""
because modify renet101 srtr,so no use pretrain 

# """
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)

model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch=0
# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            s,s1,s2,s3= model(images)
            sal_loss1 = CE(s,gts) + iou_loss(s,gts)
            sal_loss2 = CE(s1, gts) + iou_loss(s1, gts)
            sal_loss3 = CE(s2, gts) + iou_loss(s2, gts)
            sal_loss4 = CE(s3, gts) + iou_loss(s3, gts)

            sal_loss = sal_loss1 + sal_loss2 + sal_loss3 + sal_loss4
            loss = sal_loss
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || sal_loss:{:4f}  || sal_loss1:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data,sal_loss1.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} || sal_loss1:{:4f} , mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                           sal_loss.data, sal_loss1.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch != 0:
            torch.save(model.state_dict(), save_path + 'MyNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'MyNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res,s1,s2,s3= model(image)
            res= F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))



if __name__ == '__main__':
    print("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
