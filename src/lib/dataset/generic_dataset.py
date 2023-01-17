from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict

import pycocotools.coco as coco
import torch
import torch.utils.data as data

# 需要在utils文件夹下的image.py里面找
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy

'''
datasets中这些类的，这些数据集都是继承了generic_dataset，
然后他们的__get_item()部分都是直接继承了generic_dataset的
'''
class GenericDataset(data.Dataset):
  is_fusion_dataset = False
  default_resolution = None
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  # _eig_val, _eig_vec是用于进行颜色增强的
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 
    4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
    
    if ann_path is not None and img_dir is not None:
      print('==> initializing {} data from {}, \n images from {} ...'.format(
        split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path)
      self.images = self.coco.getImgIds()

      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        print('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir

  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)

    height, width = img.shape[0], img.shape[1] # height, width对应着原始图片的大小
    # 确定图像的中心点，以及s, s是和尺度相关的，目前来看就是h和w中最大的哪一个
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0
    if self.split == 'train':
      c, aug_s, rot = self._get_aug_param(c, s, width, height) # 获取仿射变化的参数
      s = s * aug_s # 原始的长或宽再乘上一个比例，进行仿射变化
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, width)

    # 仿射变化，一般不涉及到rot，主要就是利用中心点c，尺度变化s
    # 在这一步通过仿射变化从图片的原始输入尺寸变化为opt.input_w指定的输入尺寸
    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    # _get_input已经是仿射变化以后的图片了，而且也变成了c,h,w的形式
    # inp经过了仿射变化和通道数的调整
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    pre_cts, track_ids = None, None
    if opt.tracking:
      # 如果是利用单张的图片进行训练，video_id其实就对应着img_id, 然后每张图片的frame_id都为1
      # 在fake_video_data里面实现的
      pre_image, pre_anns, frame_dist = self._load_pre_data( # static image的情况下就是把当前的图片再重新的加载一次
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
      if opt.same_aug_pre and frame_dist != 0: # 如果frame_dist不为0，那么每次就都是相同的仿射变化
        trans_input_pre = trans_input 
        trans_output_pre = trans_output
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param( # 针对pre_img的时候会给center一个偏移，同时重新调整下scale
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre
        trans_input_pre = get_affine_transform( # 针对pre_img的仿射变化
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])
      pre_img = self._get_input(pre_image, trans_input_pre) # 利用得到的新的仿射变化去再次读取原本的图片
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre)
      ret['pre_img'] = pre_img # 利用新的仿射变化重新读取的
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm # pre_hm重新画的，当然画的时候也考虑了lambda_jt, lambda_fp, lambda_fn
    
    ### init samples
    self._init_ret(ret, gt_det)
    calib = self._get_calib(img_info, width, height)
    
    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      # k对应的图片中obj的索引，第k个object
      ann = anns[k] # anns对应的是current frame的
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      self._add_instance( # 这个好像是用来得到ground_truth
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
        calib, pre_cts, track_ids)

    if self.opt.debug > 0:
      gt_det = self._format_gt_det(gt_det) # 这个gt_det只是在debug的时候用的吗
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'flipped': flipped}
      ret['meta'] = meta
    return ret


  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path

  # 对于static image来说，这个函数充其量就是把原来的图片再重新的加载一次
  # training的时候是选择当前frames附近的frame就可以了
  # test的时候是要严格的满足是previous frame
  def _load_pre_data(self, video_id, frame_id, sensor_id=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previoud" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      # 为了得到img_ids这么一个元组，对于static image的训练，frame_id肯定是总为1
      # train的时候是只要和当前frame_id 小于 max_frame_dist就可以了
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
          (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else: # test的时候需要frame_dist是严格的-1，只能错开一帧
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    rand_id = np.random.choice(len(img_ids)) # 随机生成一个id,但是对于静止图片，len(img_ids) = 1
    img_id, pre_frame_id = img_ids[rand_id] # pre_frame_id肯定是总是为1的
    frame_dist = abs(frame_id - pre_frame_id) # static image, frame_dist肯定为0的
    img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir) #img_id应该就是对应coco数据集里面图片的id
    return img, anns, frame_dist


  def _get_pre_dets(self, anns, trans_input, trans_output):
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    down_ratio = self.opt.down_ratio # 目前只支持4
    trans = trans_input # 仿射变化矩阵
    reutrn_hm = self.opt.pre_hm
    # 需要注意下输入到网络里面的heatmap通道数为1，大小是和opt.input_h, opt.input_w相同的
    # 也就是说这个是利用网络输出的结果重新画的，并不是直接获取的网络的输出
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
    for ann in anns:
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -99 or \
         ('iscrowd' in ann and ann['iscrowd'] > 0):
        continue
      bbox = self._coco_box_to_bbox(ann['bbox']) # 从xywh的形式转换为xyxy的形式，标注的结果
      bbox[:2] = affine_transform(bbox[:2], trans) # 坐标点同样进行仿射变化
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1) # 防止仿射变化以后的结果越界
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0] # 仿射变化以后的h和w
      max_rad = 1
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w))) # 解方程求取的可以接受的高斯函数半径
        radius = max(0, int(radius)) 
        max_rad = max(max_rad, radius)
        ct = np.array( # 计算center
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct0 = ct.copy()
        conf = 1

        # center会有一个偏移，注意下这里对应的其实都是box的center了
        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w # lambda_jt 给轨迹加上一个小的抖动
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0 # lambda_fn有一定的几率中心点会丢掉center， conf=0

        # ct_int在加上了一点高斯抖动的 lambda_jt
        # pre_cts是用来干什么的，暂时不清楚
        ct_int = ct.astype(np.int32)
        if conf == 0:
          pre_cts.append(ct / down_ratio)
        else:
          pre_cts.append(ct0 / down_ratio)

        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if reutrn_hm: # 不管到底画不画，但是画的都是ct_int这个添加过抖动以后的center
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf) # k=conf=0的就是随机丢弃的center,lambda_fn不会在heatmap上画出

        if np.random.random() < self.opt.fp_disturb and reutrn_hm: # lambda_fp是随机的在ground-truth附近弄一个假的峰值
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32) # 这个ct2只会在heatmap上画出来，但是并不会加入到pre_cts里面
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf) # 如果是k=0的还是不会在heatmap上画出

    return pre_hm, pre_cts, track_ids #加载的都是同一张图片，对应的track_ids肯定是相同的

  def _get_border(self, border, size):
    i = 1
    # 如果图像宽（或者高） 小于等于2xboder, 则i增大为2， 返回border // i
    # 否则， 如果图像宽（或高） 大于2 x border, 则i不变，返回border
    while size - border // i <= border // i: # size < 2 * border
        i *= 2
    return border // i

  # 对于当前帧加载进来的图片得到仿射变化所必须的参数，一般disturb=False
  # 对于pre_img，调用的时候利用的是disturb=True，之前的center和s想比inp的时候都会有所变化
  def _get_aug_param(self, c, s, width, height, disturb=False):
    if (not self.opt.not_rand_crop) and not disturb:
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1)) # 得到一个随机的尺度缩放比例
      w_border = self._get_border(128, width)
      h_border = self._get_border(128, height)
      # _get_border为了保证low < high
      # 在low和high之间重新的去取center
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale #scale和shift默认的是0.1和0.4
      cf = self.opt.shift
      if type(s) == float:
        s = [s, s]
      c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf) # 给center一个偏移
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf) # 缩放比例也会有一个偏移
    
    if np.random.random() < self.opt.aug_rot: # 有一定的概率进行rotate层面的数据增强
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot # 返回仿射变化的必要参数


  def _flip_anns(self, anns, width):
    for k in range(len(anns)):
      bbox = anns[k]['bbox']
      anns[k]['bbox'] = [
        width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
      
      if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
        keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
          self.num_joints, 3)
        keypoints[:, 0] = width - keypoints[:, 0] - 1
        for e in self.flip_idx:
          keypoints[e[0]], keypoints[e[1]] = \
            keypoints[e[1]].copy(), keypoints[e[0]].copy()
        anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

      if 'rot' in self.opt.heads and 'alpha' in anns[k]:
        anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                           else - np.pi - anns[k]['alpha']

      if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
        anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

      if self.opt.velocity and 'velocity' in anns[k]:
        anns[k]['velocity'] = [-10000, -10000, -10000]

    return anns


  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    return inp


  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    # 这3个参数的大小都是根据最多的object来的max_objs
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4, 
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2, 
      'dep': 1, 'dim': 3, 'amodel_offset': 2}

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})


  def _get_calib(self, img_info, width, height):
    if 'calib' not in img_info: # 解决coco字段里面calib对应空的问题，要不然直接去除calib字段
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _ignore_region(self, region, ignore_val=1):
    np.maximum(region, ignore_val, out=region)


  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    if cls_id == 0: # ignore all classes
      self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      self._ignore_region(ret['hm'][abs(cls_id) - 1, 
                                    int(bbox[1]): int(bbox[3]) + 1, 
                                    int(bbox[0]): int(bbox[2]) + 1])
    if ('hm_hp' in ret) and cls_id <= 1:
      self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                          int(bbox[0]): int(bbox[2]) + 1])

  # xywh -> xyxy
  # 原本的应该是xywh, 这样在相加以后就得到了左上角和右下角的顶点了
  # coco默认标注的是xywh的形式
  # 我是用下面这个把csv转成coco的，他在转的时候把csv标注的xyxy -> xywh
  # https://github.com/hu64/SpotNet/blob/master/object%20detection/src/tools/convert_csv_to_coco.py
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  # 对boundingbox也进行一个输出层面上的仿射变化
  # bbox_amodal这个里面返回的好像就是没有考虑越界的bounding box
  def _get_bbox_output(self, bbox, trans_output, height, width):
    bbox = self._coco_box_to_bbox(bbox).copy() # xywh -> x1y1x2y2

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    # 对boundingbox进行同样的仿射变化
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    # 仿射变化以后重新取x1，y1，x2，y2
    # example, x1肯定要取仿射变化以后矩形左侧的最小值
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    #https://medium.com/aimmosubscribe/review-visibility-guided-nms-efficient-boosting-of-amodal-object-detection-in-crowded-traffic-e95ebb1470fc
    #关于amodal的解释
    #这样来看其实就是没有进行越界处理的boundingbox ??
    bbox_amodal = copy.deepcopy(bbox)
    # 防止越界
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    # 重新计算出wh，因为最后是要回归wh的
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal

  # k其实就是标记的索引，wh_mask，这些是不是只有在maxobjects数小于实际的object才有用，通常情况一张图一个object，那也就是第一个索引为1
  # ret['masak']这个里面这么多的mask起到的是什么作用
  # 因为对boundingbox进行了仿射变化，有些bbox可能就失效了，所以这些mask其实也就是对应着一张图片中仿射变化以后仍然有效的那些obj
  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
    aug_s, calib, pre_cts=None, track_ids=None):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    # 前面的仿射变化可能造成hw为0的情况，所以就直接跳过了
    if h <= 0 or w <= 0:
      return
    # 通过解方程来确定heatmap中center的半径
    # https://cloud.tencent.com/developer/article/1669896
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    # ct是重新利用boundingbox算出来的，这个是求ground-truth的时候
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1 # 好多类别都有个对应的mask这个到底是什么意思
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    # 这里的ind只能是把图片拉平成一维以后的位置吧
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0] # ind记录的应该是2维图片拉平以后的索引
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    # heatmap是在相应的类别所对应的通道上去画高斯圆形
    # 但是作为输入的时候就是直接在一张单通道的图上面画的
    # 这里是在类别对应的通道上去画出center了
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius) #hm在debug的时候view_as_array可以直接看到

    gt_det['bboxes'].append( # 转换维xyxy的形式
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct) # current frame的center

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int # 这个就是网络要去预测的offset
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))

    if 'ltrb' in self.opt.heads: # 计算center到四条边的距离
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    # _get_bbox_output中得到bbox_amodal
    # 好像就是没有进行越界处理的的bbox, xyxy可能处于图像外面
    if 'ltrb_amodal' in self.opt.heads:
      # 如果bbox的点处于图像外面，还是照样用center去减去，得到ltrb
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    # 似乎是需要标注里面提供vel
    if 'velocity' in self.opt.heads:
      if ('velocity' in ann) and min(ann['velocity']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])
    
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])
    
  # 应该是和位姿估计相关的，和需求的目标跟踪没什么关联
  def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
    num_joints = self.num_joints
    pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
        if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)
    if self.opt.simple_radius > 0:
      hp_radius = int(simple_radius(h, w, min_overlap=self.opt.simple_radius))
    else:
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))

    for j in range(num_joints):
      pts[j, :2] = affine_transform(pts[j, :2], trans_output)
      if pts[j, 2] > 0:
        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
          pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
          ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
          ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
          pt_int = pts[j, :2].astype(np.int32)
          ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
          ret['hp_ind'][k * num_joints + j] = \
            pt_int[1] * self.opt.output_w + pt_int[0]
          ret['hp_offset_mask'][k * num_joints + j] = 1
          ret['hm_hp_mask'][k * num_joints + j] = 1
          ret['joint'][k * num_joints + j] = j
          draw_umich_gaussian(
            ret['hm_hp'][j], pt_int, hp_radius)
          if pts[j, 2] == 1:
            ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
            ret['hp_offset_mask'][k * num_joints + j] = 0
            ret['hm_hp_mask'][k * num_joints + j] = 0
        else:
          pts[j, :2] *= 0
      else:
        pts[j, :2] *= 0
        self._ignore_region(
          ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1, 
                          int(bbox[0]): int(bbox[2]) + 1])
    gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

  def _add_rot(self, ret, ann, k, gt_det):
    if 'alpha' in ann:
      ret['rot_mask'][k] = 1
      alpha = ann['alpha']
      if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
        ret['rotbin'][k, 0] = 1
        ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)    
      if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
        ret['rotbin'][k, 1] = 1
        ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
      gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
    else:
      gt_det['rot'].append(self._alpha_to_8(0))

  # 涉及到pi的，应该是和旋转相关的
  def _alpha_to_8(self, alpha):
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
  
  def _format_gt_det(self, gt_det):
    if (len(gt_det['scores']) == 0):
      gt_det = {'bboxes': np.array([[0,0,1,1]], dtype=np.float32), 
                'scores': np.array([1], dtype=np.float32), 
                'clses': np.array([0], dtype=np.float32),
                'cts': np.array([[0, 0]], dtype=np.float32),
                'pre_cts': np.array([[0, 0]], dtype=np.float32),
                'tracking': np.array([[0, 0]], dtype=np.float32), # tracking的损失应该就是offset
                'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                'hps': np.zeros((1, 17, 2), dtype=np.float32),}
    gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
    return gt_det

  # 如果输入的不是视频，通过这个伪造成视频模式
  # 其实就是每张图片都对应一个video_id，也就是说每张图片都对应一个独立的视频
  # 然后这个视频里面就一张图片，所以frame_id永远是1
  # 同时添加了track_id，好像是根据所有的标注来分配的
  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return
    # 对于static的image来说，是第几个annotation，他的tracking_id就对应是多少
    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1 # 对于static image，
