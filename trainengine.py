# coding=utf-8
import cv2,os,logging,argparse
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import numpy as np
from time import localtime
from timeit import default_timer as timer

from net_gluon import G1_Net,G2_Net,D_Net
from srd_dataloader import *

class SRTrainEngine(object):
    def __init__(
        self,
        ctx,
        logger,
        **kwargs):
        super(SRTrainEngine, self).__init__()
        self.g1 = G1_Net(prefix='g1-')
        self.g2 = G2_Net(prefix='g2-')
        self.d1 = D_Net(prefix='d1-')
        self.d2 = D_Net(prefix='d2-')
        self.ctx = ctx
        self.logger = logger
        
        # loss weight
        self.lambda_d1 = 0.1
        self.lambda_d2 = 0.1
        self.lambda_mask = 1.0
        self.lambda_deshadow = 5.0
        # loss
        self.d1_loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=self.lambda_d1)
        self.d2_loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=self.lambda_d2)
        self.mask_l1_loss = gluon.loss.L1Loss(weight=self.lambda_mask)
        self.deshadow_l1_loss = gluon.loss.L1Loss(weight=self.lambda_deshadow)

        self.metric_train = mx.metric.create(
            [
                mx.metric.Accuracy(name='mask_acc_real'),
                mx.metric.Accuracy(name='deshadow_acc_real'),
                mx.metric.Accuracy(name='mask_acc_fake'),
                mx.metric.Accuracy(name='deshadow_acc_fake'),
                mx.metric.MAE(name='mask_mae'),
                mx.metric.MAE(name='deshadow_mae')
            ]
        )
        self.metric_eval = mx.metric.create(
            [
                mx.metric.MAE(name='mask_mae'),
                mx.metric.MAE(name='deshadow_mae')
            ]
        )
        self.mask_cls_metric = mx.metric.Accuracy()
        self.gt_cls_metric = mx.metric.Accuracy()
        self.mask_reg_metric = mx.metric.MAE()
        self.deshadow_reg_metric = mx.metric.MAE()
        return

    def export_symbol(self, output_fn):
        with self.g1.name_scope:
            self.g1(mx.sym.Variable('data')).save(output_fn+'-g1')
        with self.g2.name_scope:
            self.g2(mx.sym.Variable('data')).save(output_fn+'-g2')

    def save_checkpoint(self, e, modelname, epoch_check_period, if_save_states=False):
        return 

    def batch_train(self, triplet, trainer_quard):
        src_shadow_batch, src_mask_batch, src_gt_batch = triplet
        src_shadow_batch_ls = gluon.utils.split_and_load(src_shadow_batch, self.ctx)
        src_mask_batch_ls = gluon.utils.split_and_load(src_mask_batch, self.ctx)
        src_gt_batch_ls = gluon.utils.split_and_load(src_gt_batch, self.ctx)

        g1_trainer, g2_trainer, d1_trainer, d2_trainer = trainer_quard

        batch_size = src_shadow_batch_ls[0].shape[0]
        label_ones = [mx.nd.ones((x.shape[0],),y) for x,y in zip(src_shadow_batch_ls, self.ctx)]
        label_zeros = [mx.nd.zeros_like(x) for x in label_ones]
        
        with ag.record():
            # update D: maximize log(D(x)) + log(1 - D(G(z)))
            # real
            real_d1_out_ls = [self.d1(x,y) for x, y in zip(src_shadow_batch_ls,src_mask_batch_ls)]
            real_d2_out_ls = [self.d2(x,y,z) for x,y,z in zip(src_shadow_batch_ls, src_mask_batch_ls, src_gt_batch_ls)]
            errD1_real_ls = [self.d1_loss(x, y) for x, y in zip(real_d1_out_ls, label_ones)]
            errD2_real_ls = [self.d2_loss(x,y) for x, y in zip(real_d2_out_ls, label_ones)]
            # fake
            fake_g1_out_ls = [self.g1(x) for x in src_shadow_batch_ls]
            fake_d1_out_ls = [self.d1(x,y) for x, y in zip(src_shadow_batch_ls, fake_g1_out_ls)]
            fake_g2_out_ls = [self.g2(x,y) for x,y in zip(src_shadow_batch, fake_g1_out_ls)]
            fake_d2_out_ls = [self.d2(x,y,z) for x,y,z in zip(fake_g1_out_ls, fake_g2_out_ls, src_gt_batch_ls)]
            errD1_fake_ls = [self.d1_loss(x, y) for x, y in zip(fake_d1_out_ls, label_zeros)]
            errD2_fake_ls = [self.d2_loss(x,y) for x, y in zip(fake_d2_out_ls, label_zeros)]
            reg_mask_ls = [self.mask_l1_loss(x,y) for x,y in zip(fake_g1_out_ls, src_mask_batch_ls)]
            reg_deshadow_ls = [self.deshadow_l1_loss(x,y) for x,y in zip(fake_g2_out_ls, src_gt_batch_ls)]
            total_error = sum(errD1_real_ls) + sum(errD2_real_ls) + sum(errD1_fake_ls) + sum(errD2_fake_ls) + sum(reg_mask_ls) + sum(reg_deshadow_ls)
            total_error.backward()
            # update G: maximize log(D(G(z)))


            # metric
            # real
            for p,q,r,s,t,u,v,w in zip(
                real_d1_out_ls,
                real_d2_out_ls,
                fake_d1_out_ls,
                fake_d2_out_ls,
                fake_g1_out_ls,
                fake_g2_out_ls,
                src_mask_batch_ls,
                src_gt_batch_ls
                ):
                self.metric_train.update(
                    labels=[
                        label_ones,
                        label_ones,
                        label_zeros,
                        label_zeros,
                        v, # src_mask_batch
                        w, # src_gt_batch
                        ],
                    preds=[
                        p, # real_d1_out
                        q, # real_d2_out
                        r, # fake_d1_out_ls
                        s, # fake_d2_out_ls
                        t, # fake_g1_out_ls
                        u, # fake_g2_out_ls
                        ])


        return 

    def batch_eval(self, triplet):
        return

    def eval_process(self, val_data_loader):
        return

    def _batch_period_call(self, e, i, check_period, lastcheck):
        return

    def _epoch_period_call(self, e, modelname, epoch_check_period, if_save_states=False):
        return