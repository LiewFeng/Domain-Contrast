# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
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

import cv2

from collections import defaultdict

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.label_file import LabelFile
from model.utils.parser_func import parse_args, set_dataset_args
clipart_labels = [
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
water_labels = [#'__background__',  # always index 0
                         'bicycle', 'bird', 'car', 'cat', 'dog', 'person']
cityscape_labels = [#'__background__',  # always index 0
                         'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
sim10k_labels = [#'__background__',  # always index 0
                          'car']



try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3




lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  
  args = set_dataset_args(args)

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, False)
  imdb.competition_mode(on=True)
  
  args.labels = imdb.classes
  
  print('{:d} roidb entries'.format(len(roidb)))

#   input_dir = args.load_dir + "/" + args.net + "/" + args.dataset + "2VOC07"
  input_dir = args.load_dir + "/" + args.net + "/" + args.log_ckpt_name
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'contrastive_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
#   load_name = os.path.join(input_dir,
#     'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
#   load_name = os.path.join(input_dir,
#     'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
#   tmp_dir = '/userhome/Faster-RCNN/faster-rcnn1/models/res101/pascal_voc_0712'
#   load_name = os.path.join(tmp_dir,
#     'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
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

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  num_images = len(imdb.image_index)
#   print('image_index: {}'.format(imdb.image_index))
#   pdb.set_trace()

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}

  fasterRCNN.eval()
  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()
      
#       print('scores.shape: {}'.format(scores.shape))
      scores = scores.squeeze()
#       print('scores[0]: {}'.format(scores[0]))
#       print('scores[0].sum(): {}'.format(scores[0].sum()))
#       scores = nn.Softmax(dim=-1)(scores)
#       print('scores[0]: {}'.format(scores[0]))
#       pdb.set_trace()
#       print('scores: {}'.format(scores))
#       print('scores[0].sum(): {}'.format(scores[0].sum()))
#       pdb.set_trace()
      pred_boxes = pred_boxes.squeeze()
      
      proper_dets = defaultdict(list)
#       PL_thresh = 0.99
      for j in xrange(1, imdb.num_classes):          
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
#             cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
      
            # select high confidence box
            mask = cls_dets[:, -1].gt(args.conf_thresh).expand(5, cls_dets.size(0)).t()
            cls_dets = torch.masked_select(cls_dets, mask).view(-1, 5)  # delete boxes with score < args.conf_thresh
            if cls_dets.size(0) == 0:
                continue
            tmp_boxes = cls_dets[:, :4]
            for idx in range(tmp_boxes.size(0)):
              proper_dets[args.labels[j]].append(tmp_boxes[idx])
            
#             tmp_boxes = cls_dets[:, :4]
#             tmp_scores = cls_dets[:, -1].cpu().numpy()
#             ind  = np.argsort(tmp_scores)[-1]
# #             print('tmp_scores: {}'.format(tmp_scores))
# #             print('tmp_scores[ind]: {}'.format(tmp_scores[ind]))
# #             pdb.set_trace()
#             box = tmp_boxes[ind]
#             proper_dets[args.labels[j-1]].append(box)
            
      img_id = imdb.image_index[i] 
#       ids.append(str(img_id) + '\n')
      filename = os.path.join(args.save_dir, 'Annotations', str(img_id) + '.xml')
      img_path = os.path.join(args.save_dir, 'JPEGImages', str(img_id) + '.jpg')
      labeler = LabelFile(filename, img_path, args.labels)
      labeler.savePascalVocFormat(proper_dets)
      print('Saved to {:s}'.format(filename))

#             print('boxes.shape: {}'.format(boxes.shape))
#             print('scores.shape: {}'.format(scores.shape))
#             print('scores: {}'.format(scores))
#             pdb.set_trace()




#   end = time.time()
#   print("test time: %0.4fs" % (end - start))
