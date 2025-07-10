import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from model.MyNet import MyNet as MALNet
from data_cod import test_dataset
import time
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./Test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = MALNet()
model.load_state_dict(torch.load('./model/Net.pth'))
model.cuda()
model.eval()
# test
test_datasets = ['EORSSD']

costtim = []

for dataset in test_datasets:
    save_path = './pre_map/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    cost_time = []
    mae_sum = 0
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start_time = time.perf_counter()
        res,s,s1,s2= model(image)
        cost_time.append(time.perf_counter()-start_time)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res*255)
        mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    mae = mae_sum / test_loader.size
    cost_time.pop(0)
    print(dataset,' MAE : ',mae)
    print('Mean running time is :',np.mean(cost_time))
    print("FPS is :",test_loader.size/np.sum(cost_time))

