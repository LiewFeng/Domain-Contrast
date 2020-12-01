from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.contrastive_loss import ContrastiveLoss

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from model.utils.plt_loss import plt_loss

from model.utils.parser_func import parse_args, set_dataset_args


class DualDataset(Dataset):
    def __init__(self, set1, set2):
        super(DualDataset, self).__init__()
        self.set1 = set1
        self.set2 = set2

    def __getitem__(self, item):
        return self.set1[item], self.set2[item]

    def __len__(self):
        return len(self.set1)
      
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    # for contrastive loss
#     rand_num = torch.arange(0,self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  args = set_dataset_args(args)
    
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
#   cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  # source dataset
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)
  # target dataset
  imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
  train_size_t = len(roidb_t)

  print('{:d} source roidb entries'.format(len(roidb)))
  print('{:d} target roidb entries'.format(len(roidb_t)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.log_ckpt_name
#   output_dir = args.save_dir + "/" + args.net + "/" + args.dataset_t
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)
  sampler_batch_t = sampler(train_size_t, args.batch_size)

  dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

#   dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
#                                                sampler=sampler_batch, num_workers=args.num_workers)
  dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
#   dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
#                                                sampler=sampler_batch_t, num_workers=args.num_workers)
  dataloader = torch.utils.data.DataLoader(
      DualDataset(dataset_s, dataset_t),
      batch_size=args.batch_size,
      sampler=sampler(train_size, args.batch_size),
      num_workers=args.num_workers,
      pin_memory=True
  )
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
  conrastive_criterion = ContrastiveLoss()
  if args.resume:
    load_name = os.path.join(output_dir,
      'contrastive_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
#     load_name = os.path.join(output_dir,
#       'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
#     args.session = checkpoint['session']
#     args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     lr = optimizer.param_groups[0]['lr']
#     optimizer.param_groups[0]['lr'] = args.lr
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
  
  loss1 = []
    
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

#     if epoch % (args.lr_decay_step + 1) == 0:
    if (epoch-1) in args.lr_decay_step:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

#         data_iter_s = iter(dataloader_s)
#         data_iter_t = iter(dataloader_t)

    
#     for step in range(iters_per_epoch):
#       try:
#         data_s = next(data_iter_s)
#       except:
#         data_iter_s = iter(dataloader_s)
#         data_s = next(data_iter_s)
#       try:
#         data_t = next(data_iter_t)
#       except:
#         data_iter_t = iter(dataloader_t)
#         data_t = next(data_iter_t)

    img_loss = 0
    
    for step, (data_s, data_t) in enumerate(iter(dataloader)):
      # source domain
      with torch.no_grad():
              im_data.resize_(data_s[0].size()).copy_(data_s[0])
              im_info.resize_(data_s[1].size()).copy_(data_s[1])
              gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
              num_boxes.resize_(data_s[3].size()).copy_(data_s[3])
      if im_data.size(0) != args.batch_size:
        continue
        
#       print('im_data.shape: {}'.format(im_data.shape))
#       print('gt_boxes[0]: {}'.format(gt_boxes[0]))
      
      fasterRCNN.zero_grad()
      feat_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, mode=args.mode)
      # target domain
      with torch.no_grad():
              im_data.resize_(data_t[0].size()).copy_(data_t[0])
              im_info.resize_(data_t[1].size()).copy_(data_t[1])
              gt_boxes.resize_(data_t[2].size()).copy_(data_t[2])
              num_boxes.resize_(data_t[3].size()).copy_(data_t[3])
              
#       print('num_boxes_t: {}'.format(num_boxes))
#       print('gt_boxes[0]: {}'.format(gt_boxes[0]))
#       pdb.set_trace()

      feat_t = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, mode=args.mode)
      contrastive_loss = conrastive_criterion(feat_s.view(feat_s.size(0), -1), 
                                              feat_t.view(feat_t.size(0), -1), 
                                              t=args.t)
#       if args.mode == 'image_level':
#         contrastive_loss = conrastive_criterion(feat_s.view(args.batch_size, -1).contiguous(), 
#                                                 feat_t.view(args.batch_size, -1).contiguous(), 
#                                                 t=args.t)
#       else:
#         contrastive_loss = conrastive_criterion(feat_s, 
#                                                 feat_t, 
#                                                 t=args.t)
      loss = contrastive_loss * args.lambd
      loss_temp += loss.item()
        

      img_loss += contrastive_loss.mean().item()
      
      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_contrastive = contrastive_loss.mean().item()
        else:
          loss_contrastive = contrastive_loss.item()

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\ttime cost: %f" % (end-start))
        print("\t\t\tloss_contrastive: %.4f" % (loss_contrastive))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()
        
    loss1.append(img_loss/step)
    plt_loss(epoch, 'output/{}'.format(args.dataset_t), 'Image Level', loss1)
    
    save_name = os.path.join(output_dir, 'contrastive_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
