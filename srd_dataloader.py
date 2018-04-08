# coding=utf-8
import os,cv2
from random import uniform, seed, randint
from time import time
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

import numpy as np

seed(time())
class TripletAug(object):
    def __init__(self, augsize=256, random_warp_ratio=0.1,):
        super(TripletAug, self).__init__()
        self.augsize=augsize
        self.random_warp_ratio = random_warp_ratio

    def _getM(self, shadow_img):
        h,w = shadow_img.shape[0], shadow_img.shape[1]
        src_points = np.array([ (w * uniform(0,self.random_warp_ratio), h * uniform(0,self.random_warp_ratio)),
                                (w - w * uniform(0,self.random_warp_ratio), h * uniform(0,self.random_warp_ratio)),
                                (w * uniform(0,self.random_warp_ratio), h - h * uniform(0,self.random_warp_ratio)),
                                (w - w * uniform(0,self.random_warp_ratio), h - h * uniform(0,self.random_warp_ratio))],
                                dtype=np.float32)
        dst_points = np.array([(0,0),(w - 1,0),(0,h-1),(w-1, h-1)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M

    def _perspective_jitter(self, src_img, M):
        w_h = (src_img.shape[1], src_img.shape[0])
        warped_img = cv2.warpPerspective(src_img.asnumpy(), M, w_h, flags = cv2.INTER_CUBIC)
        return warped_img

    def _resize(self, src):
        return mx.nd.array(cv2.resize(src,(self.augsize, self.augsize),interpolation=cv2.INTER_CUBIC), mx.cpu())

    def __call__(self, triplet):
        shadow_img, mask_img, gt_img = triplet
        shadow_img = mx.image.resize_short(src=shadow_img,size=self.augsize)
        mask_img = mx.image.resize_short(src=mask_img,size=self.augsize)
        gt_img = mx.image.resize_short(src=gt_img,size=self.augsize)
        # get M
        M = self._getM(shadow_img)
        pj_shadow_img = self._perspective_jitter(src_img=shadow_img, M=M)
        pj_shadow_img = self._resize(pj_shadow_img)
        pj_mask_img = self._perspective_jitter(src_img=mask_img, M=M)
        pj_mask_img = self._resize(pj_mask_img)
        pj_gt_img = self._perspective_jitter(src_img=gt_img, M=M)
        pj_gt_img = self._resize(pj_gt_img)
        return pj_shadow_img, pj_mask_img, pj_gt_img

class DatasetForSRD(gluon.data.dataset.Dataset):
    def __init__(self, src_dir_triplet, short_size=320):
        super(DatasetForSRD, self).__init__()
        self.short_size = short_size
        self.triplet_original_img = []
        self.instance_num = 0
        shadow_dir, mask_dir, gt_dir = src_dir_triplet
        shadow_fn_ls, mask_fn_ls, gt_fn_ls = self._check_all(shadow_dir, mask_dir, gt_dir)
        for shadow_fn, mask_fn, gt_fn in zip(shadow_fn_ls, mask_fn_ls, gt_fn_ls):
            shadow_img = mx.image.imread(os.path.join(shadow_dir, shadow_fn))
            shadow_img = mx.image.resize_short(src=shadow_img, size=self.short_size)
            mask_img = mx.image.imread(os.path.join(mask_dir, mask_fn))
            mask_img = mx.image.resize_short(src=mask_img, size=self.short_size)
            gt_img = mx.image.imread(os.path.join(gt_dir, gt_fn))
            gt_img = mx.image.resize_short(src=gt_img, size=self.short_size)
            self.triplet_original_img.append((shadow_img, mask_img, gt_img))
        self.instance_num = len(self.triplet_original_img)
        return

    def _check_all(self, shadow_dir, mask_dir, gt_dir):
        shadow_fn_ls = os.listdir(shadow_dir)
        mask_fn_ls = os.listdir(mask_dir)
        gt_fn_ls = os.listdir(gt_dir)
        shadow_fn_ls.sort()
        mask_fn_ls.sort()
        gt_fn_ls.sort()
        if_valid_triplet_func = lambda x:(
            x in mask_fn_ls and 
            x in gt_fn_ls and 
            cv2.imread(os.path.join(shadow_dir, x)) is not None and 
            cv2.imread(os.path.join(mask_dir, x)) is not None and 
            cv2.imread(os.path.join(gt_dir, x)) is not None
        )
        verified_shadow_fn_ls = filter(if_valid_triplet_func, shadow_fn_ls)
        if_valid_func = lambda x:x in verified_shadow_fn_ls
        verified_mask_fn_ls = filter(if_valid_func, mask_fn_ls)
        verified_gt_fn_ls = filter(if_valid_func, gt_fn_ls)
        assert(len(verified_shadow_fn_ls) == len(verified_mask_fn_ls) == len(verified_gt_fn_ls))
        return verified_shadow_fn_ls, verified_mask_fn_ls, verified_gt_fn_ls

    def __getitem__(self, idx):
        return self.triplet_original_img[idx]

    def __len__(self):
        return self.instance_num

class BatchfyFunc4SRD(object):
    def __init__(self, size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), aug=TripletAug(), **kwargs):
        super(BatchfyFunc4SRD, self).__init__()
        if type(size) is int:
            self.wh = (size,size)
        elif type(size) is tuple or type(size) is list:
            assert(type(size[0]) is int and type(size[1]) is int)
            self.wh = (size[0], size[1])
        else:
            raise ValueError('Inapporopriate value for size')
        self.aug = aug
        self.mean = mean
        self.std = std
        return
    
    
    def _resize(self, src):
        return mx.nd.array(cv2.resize(src, self.wh, interpolation=cv2.INTER_CUBIC), mx.cpu())

    def _norm_demean_dedev(self, src_triplet):
        tensor_triplet = map(mx.nd.image.to_tensor, src_triplet)
        normed_tensor_triplet = map(lambda x:mx.nd.image.normalize(x, mean=self.mean, std=self.std) , tensor_triplet) 
        return normed_tensor_triplet
    
    def preprocess(self, src_triplet):
        if self.aug is None:
            rsz_triplet = map(self._resize, src_triplet)
        else:
            rsz_triplet= self.aug(src_triplet)
        out_img = self._norm_demean_dedev(rsz_triplet)
        return out_img
    
    def __call__(self, data):
        triplet_ls = map(self.preprocess, data)
        shadow_ls = [x[0] for x in triplet_ls]
        shadow_batch = mx.nd.stack(*shadow_ls)
        mask_ls = [x[1] for x in triplet_ls]
        mask_batch = mx.nd.stack(*mask_ls)
        gt_ls = [x[2] for x in triplet_ls]
        gt_batch = mx.nd.stack(*gt_ls)

        return shadow_batch, mask_batch, gt_batch

def inv_norm_demean_dedev(src, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    _std = mx.nd.reshape(mx.nd.array(std),(3,1,1))
    _mean = mx.nd.reshape(mx.nd.array(mean),(3,1,1))
    inv_dev_img = src *_std + _mean
    inv_norm_img = np.transpose(255*inv_dev_img, axes=(1,2,0)).clip(0,255).astype(np.uint8)
    return inv_norm_img

from math import floor,sqrt
def getvisual(src, ifchannelswap=False, layout=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    inv_imgs = map(lambda x:inv_norm_demean_dedev(x,mean,std), src)
    inv_imgs = map(lambda x:mx.nd.clip(x, 0, 255), inv_imgs)

    patch_rows, patch_cols = src.shape[2], src.shape[3]
    # determin layout
    if layout is None:
        layout_cols, layout_rows = 1, src.shape[0]
    else:
        assert(layout[0] * layout[1] == src.shape[0])
        layout_rows,layout_cols = layout
    #
    out_arr = np.zeros((layout_rows * src.shape[2], layout_cols * src.shape[3], src.shape[1]), dtype=np.uint8)
    for i, inv_img in enumerate(inv_imgs):
        r = i // layout_cols
        c = i % layout_cols
        out_arr[r*patch_rows:(r+1)*patch_rows,c*patch_cols:(c+1)*patch_cols,:] = inv_img.asnumpy()
    
    if ifchannelswap:
        out_arr = cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB)
    return out_arr

if __name__ == '__main__':

    def test_triplet_aug():
        shadow_path = r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_A\1-1.png' 
        mask_path = r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_B\1-1.png' 
        gt_path = r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_C\1-1.png' 
        shadow_img = mx.image.imdecode(open(shadow_path, 'rb').read())
        mask_img = mx.image.imdecode(open(mask_path, 'rb').read())
        gt_img = mx.image.imdecode(open(gt_path, 'rb').read())
        test_aug = TripletAug()
        while 1:
            test_tri = test_aug(shadow_img, mask_img, gt_img)
            dst_img_ls = map(lambda x:cv2.cvtColor(x.asnumpy().astype(np.uint8),cv2.COLOR_BGR2RGB),test_tri)
            cv2.imshow('test0', dst_img_ls[0])
            cv2.imshow('test1', dst_img_ls[1])
            cv2.imshow('test2', dst_img_ls[2])
            q = cv2.waitKey()
            if q ==27:
                break
        cv2.destroyAllWindows()
        return

    # test_triplet_aug()

    def test_dataloader():
        # train_dir_triplet = (
        #     r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_A',
        #     r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_B',
        #     r'D:\Datasets\ISTD_Dataset_for_shadow_removel\train\train_C'
        # )
        train_dir_triplet = (
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_A',
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_B',
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_C'
        )
        test_dir_triplet = (
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_A',
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_B',
            r'D:\Datasets\ISTD_Dataset_for_shadow_removel\test\test_C'
        )
        db_train = DatasetForSRD(src_dir_triplet=train_dir_triplet, short_size=320)
        db_test = DatasetForSRD(src_dir_triplet=test_dir_triplet, short_size=320)


        batch_size=4
        aug = TripletAug()
        batchfy_func = BatchfyFunc4SRD(aug=aug)
        batchfy_func_plain = BatchfyFunc4SRD(aug=None)

        dl_train = gluon.data.dataloader.DataLoader(
            dataset=db_train,
            batch_size=batch_size,
            shuffle=True,
            batchify_fn=batchfy_func,
            last_batch='discard'
        )

        dl_plain = gluon.data.dataloader.DataLoader(
            dataset=db_test,
            batch_size=batch_size,
            shuffle=False,
            batchify_fn=batchfy_func_plain,
            last_batch='discard'
        )
        # show batch
        # cv2.namedWindow('shadow',0)
        # cv2.namedWindow('mask',0)
        # cv2.namedWindow('gt',0)
        cv2.namedWindow('shadow')
        cv2.namedWindow('mask')
        cv2.namedWindow('gt')

        for i, triplet in enumerate(dl_train):
            shadow, mask, gt = triplet
            shadow_show = getvisual(shadow, ifchannelswap=True, layout=(batch_size/4,4))
            mask_show = getvisual(mask, ifchannelswap=True, layout=(batch_size/4,4))
            gt_show = getvisual(gt, ifchannelswap=True, layout=(batch_size/4,4))

            cv2.imshow('shadow', shadow_show)
            cv2.imshow('mask', mask_show)
            cv2.imshow('gt', gt_show)    

            q = cv2.waitKey()
            if q==27:
                break       
        
        for i, triplet in enumerate(dl_plain):
            shadow, mask, gt = triplet
            shadow_show = getvisual(shadow, ifchannelswap=True, layout=(batch_size/4,4))
            mask_show = getvisual(mask, ifchannelswap=True, layout=(batch_size/4,4))
            gt_show = getvisual(gt, ifchannelswap=True, layout=(batch_size/4,4))

            cv2.imshow('shadow', shadow_show)
            cv2.imshow('mask', mask_show)
            cv2.imshow('gt', gt_show)    

            q = cv2.waitKey()
            if q==27:
                break
        cv2.destroyAllWindows()
        return

    test_dataloader()