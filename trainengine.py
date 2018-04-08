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

from net_gluon import G1_Net,G2_Net,get_D_net
from srd_dataloader import *

class TRTrainEngine(object):
    def __init__(
        self,
        trainer,
        charset,
        ctx,
        logger,
        **kwargs):
        super(TRTrainEngine, self).__init__()
        self.g1 = G1_Net(prefix='g1-')
        self.g2 = G2_Net(prefix='g2-')
        self.d1 = get_D_net(prefix='d1-')
        self.d2 = get_D_net(prefix='d2-')
        self.trainer = trainer
        self.ctx = ctx
        self.logger = logger

        self.mask_cls_metric = mx.metric.Accuracy()
        self.gt_cls_metric = mx.metric.Accuracy()
        self.ft_reg_metric = mx.metric.MAE()

        return

    def export_symbol(self, output_fn):
        with self.g1.name_scope:
            self.g1(mx.sym.Variable('data')).save(output_fn+'-g1')
        with self.g2.name_scope:
            self.g2(mx.sym.Variable('data')).save(output_fn+'-g2')

    def save_checkpoint(self, e, modelname, epoch_check_period, if_save_states=False):
        return 

    def batch_step(self, triplet, ifbackward=True):
        return 

    def eval_process(self, val_data_loader):
        return

    def _batch_period_call(self, e, i, check_period, lastcheck):
        return

    def _epoch_period_call(self, e, modelname, epoch_check_period, if_save_states=False):
        return