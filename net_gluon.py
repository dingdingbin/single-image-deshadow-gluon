# coding=utf-8
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag


# designed for 256x256 input

intround = lambda x:int(round(x))
class G1_Net(nn.HybridBlock):
    def __init__(self, slope=0.2, nfactor=1.0, **kwargs):
        super(G1_Net, self).__init__(**kwargs)
        with self.name_scope():
            # Cv0
            self.cv0 = nn.HybridSequential(prefix='Cv0-')
            with self.cv0.name_scope():
                self.cv0.add(
                    nn.Conv2D(channels=intround(64* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                )
            # Cv1
            self.cv1 = nn.HybridSequential(prefix='Cv1-')
            with self.cv1.name_scope():
                self.cv1.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(128* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv2
            self.cv2 = nn.HybridSequential(prefix='Cv2-')
            with self.cv2.name_scope():
                self.cv2.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(256* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv3
            self.cv3 = nn.HybridSequential(prefix='Cv3-')
            with self.cv3.name_scope():
                self.cv3.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-0
            self.cv4_0 = nn.HybridSequential(prefix='Cv4-0-')
            with self.cv4_0.name_scope():
                self.cv4_0.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-1
            self.cv4_1 = nn.HybridSequential(prefix='Cv4-1-')
            with self.cv4_1.name_scope():
                self.cv4_1.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-2
            self.cv4_2 = nn.HybridSequential(prefix='Cv4-2-')
            with self.cv4_2.name_scope():
                self.cv4_2.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv5
            self.cv5 = nn.HybridSequential(prefix='cv5-')
            with self.cv5.name_scope():
                self.cv5.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                )
            # cvT6
            self.cvT6 = nn.HybridSequential(prefix='cvT6-')
            with self.cvT6.name_scope():
                self.cvT6.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-0
            self.cvT7_0 = nn.HybridSequential(prefix='cvT7-0-')
            with self.cvT7_0.name_scope():
                self.cvT7_0.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-1
            self.cvT7_1= nn.HybridSequential(prefix='cvT7-1-')
            with self.cvT7_1.name_scope():
                self.cvT7_1.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-2
            self.cvT7_2 = nn.HybridSequential(prefix='cvT7-2-')
            with self.cvT7_2.name_scope():
                self.cvT7_2.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT8
            self.cvT8 = nn.HybridSequential(prefix='cvT8-')
            with self.cvT8.name_scope():
                self.cvT8.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(256* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT9
            self.cvT9 = nn.HybridSequential(prefix='cvT9-')
            with self.cvT9.name_scope():
                self.cvT9.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(128* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT10
            self.cvT10 = nn.HybridSequential(prefix='cvT10-')
            with self.cvT10.name_scope():
                self.cvT10.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(64* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT11
            self.cvT11 = nn.HybridSequential(prefix='cvT11-')
            with self.cvT11.name_scope():
                self.cvT11.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=1, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.Activation('tanh')
                )
            
    def hybrid_forward(self, F, x):
        cv0_out = self.cv0(x)
        cv1_out = self.cv1(cv0_out)
        cv2_out = self.cv2(cv1_out)
        cv3_out = self.cv3(cv2_out)
        cv4_0_out = self.cv4_0(cv3_out)
        cv4_1_out = self.cv4_1(cv4_0_out)
        cv4_2_out = self.cv4_2(cv4_1_out)
        cv5_out = self.cv5(cv4_2_out)
        cvT6_out = self.cvT6(cv5_out)
        cvT7_0_in = F.concat(cvT6_out, cv4_2_out, dim=1)
        cvT7_0_out = self.cvT7_0(cvT7_0_in)
        cvT7_1_in = F.concat(cvT7_0_out, cv4_1_out, dim=1)
        cvT7_1_out = self.cvT7_1(cvT7_1_in)
        cvT7_2_in = F.concat(cvT7_1_out, cv4_0_out, dim=1)
        cvT7_2_out = self.cvT7_2(cvT7_2_in)
        cvT8_in = F.concat(cvT7_2_out, cv3_out)
        cvT8_out = self.cvT8(cvT8_in)
        cvT9_in = F.concat(cvT8_out, cv2_out)
        cvT9_out = self.cvT9(cvT9_in)
        cvT10_in = F.concat(cvT9_out, cv1_out)
        cvT10_out = self.cvT10(cvT10_in)
        cvT11_in = F.concat(cvT10_out, cv0_out)
        cvT11_out = self.cvT11(cvT11_in)
        return cvT11_out

class G2_Net(nn.HybridBlock):
    def __init__(self, slope=0.2, nfactor=1.0, **kwargs):
        super(G2_Net, self).__init__(**kwargs)
        with self.name_scope():
            # Cv0
            self.cv0 = nn.HybridSequential(prefix='Cv0-')
            with self.cv0.name_scope():
                self.cv0.add(
                    nn.Conv2D(channels=intround(64* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                )
            # Cv1
            self.cv1 = nn.HybridSequential(prefix='Cv1-')
            with self.cv1.name_scope():
                self.cv1.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(128* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv2
            self.cv2 = nn.HybridSequential(prefix='Cv2-')
            with self.cv2.name_scope():
                self.cv2.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(256* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv3
            self.cv3 = nn.HybridSequential(prefix='Cv3-')
            with self.cv3.name_scope():
                self.cv3.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-0
            self.cv4_0 = nn.HybridSequential(prefix='Cv4-0-')
            with self.cv4_0.name_scope():
                self.cv4_0.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-1
            self.cv4_1 = nn.HybridSequential(prefix='Cv4-1-')
            with self.cv4_1.name_scope():
                self.cv4_1.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv4-2
            self.cv4_2 = nn.HybridSequential(prefix='Cv4-2-')
            with self.cv4_2.name_scope():
                self.cv4_2.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # Cv5
            self.cv5 = nn.HybridSequential(prefix='cv5-')
            with self.cv5.name_scope():
                self.cv5.add(
                    nn.LeakyReLU(alpha=slope),
                    nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                )
            # cvT6
            self.cvT6 = nn.HybridSequential(prefix='cvT6-')
            with self.cvT6.name_scope():
                self.cvT6.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-0
            self.cvT7_0 = nn.HybridSequential(prefix='cvT7-0-')
            with self.cvT7_0.name_scope():
                self.cvT7_0.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-1
            self.cvT7_1= nn.HybridSequential(prefix='cvT7-1-')
            with self.cvT7_1.name_scope():
                self.cvT7_1.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT7-2
            self.cvT7_2 = nn.HybridSequential(prefix='cvT7-2-')
            with self.cvT7_2.name_scope():
                self.cvT7_2.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT8
            self.cvT8 = nn.HybridSequential(prefix='cvT8-')
            with self.cvT8.name_scope():
                self.cvT8.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(256* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT9
            self.cvT9 = nn.HybridSequential(prefix='cvT9-')
            with self.cvT9.name_scope():
                self.cvT9.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(128* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT10
            self.cvT10 = nn.HybridSequential(prefix='cvT10-')
            with self.cvT10.name_scope():
                self.cvT10.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=intround(64* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.BatchNorm()
                )
            # cvT11
            self.cvT11 = nn.HybridSequential(prefix='cvT11-')
            with self.cvT11.name_scope():
                self.cvT11.add(
                    nn.Activation('relu'),
                    nn.Conv2DTranspose(channels=3, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),
                    nn.Activation('tanh')
                )
            
    def hybrid_forward(self, F, x, mask):
        conc_src = F.concat(x,mask,dim=1)
        cv0_out = self.cv0(conc_src)
        cv1_out = self.cv1(cv0_out)
        cv2_out = self.cv2(cv1_out)
        cv3_out = self.cv3(cv2_out)
        cv4_0_out = self.cv4_0(cv3_out)
        cv4_1_out = self.cv4_1(cv4_0_out)
        cv4_2_out = self.cv4_2(cv4_1_out)
        cv5_out = self.cv5(cv4_2_out)
        cvT6_out = self.cvT6(cv5_out)
        cvT7_0_in = F.concat(cvT6_out, cv4_2_out, dim=1)
        cvT7_0_out = self.cvT7_0(cvT7_0_in)
        cvT7_1_in = F.concat(cvT7_0_out, cv4_1_out, dim=1)
        cvT7_1_out = self.cvT7_1(cvT7_1_in)
        cvT7_2_in = F.concat(cvT7_1_out, cv4_0_out, dim=1)
        cvT7_2_out = self.cvT7_2(cvT7_2_in)
        cvT8_in = F.concat(cvT7_2_out, cv3_out)
        cvT8_out = self.cvT8(cvT8_in)
        cvT9_in = F.concat(cvT8_out, cv2_out)
        cvT9_out = self.cvT9(cvT9_in)
        cvT10_in = F.concat(cvT9_out, cv1_out)
        cvT10_out = self.cvT10(cvT10_in)
        cvT11_in = F.concat(cvT10_out, cv0_out)
        cvT11_out = self.cvT11(cvT11_in)
        return cvT11_out

class D_Net(nn.HybridBlock):
    def __init__(self, slope=0.2, nfactor=1.0, **kwargs):
        super(D_Net, self).__init__(**kwargs)
        with self.name_scope():
            self.d_net = nn.HybridSequential()
            self.d_net.add(
                nn.Conv2D(channels=intround(64* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 256->128
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=intround(128* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 128->64
                nn.BatchNorm(),
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=intround(256* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 64->32
                nn.BatchNorm(),
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 32->16
                nn.BatchNorm(),
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 16->8
                nn.BatchNorm(),
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=intround(512* nfactor), kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=True),# 8->4
                nn.BatchNorm(),
                nn.LeakyReLU(alpha=slope),
                nn.Conv2D(channels=2, kernel_size=(4,4), strides=(1,1), padding=(0,0), use_bias=True),# 4->1
                nn.Flatten()
            )
        return
    def hybrid_forward(self, F, *args):
        data_con = F.concat(*args, dim=1)
        return self.d_net(data_con)

if __name__ == '__main__':
    g1 = G1_Net(prefix='g1-')
    g2 = G2_Net(prefix='g2-')

    d1 = D_Net(prefix='d1-')
    d2 = D_Net(prefix='d2-')

    print('############## g1:')
    g1_sym = g1(mx.sym.Variable('data'))
    mx.viz.print_summary(g1_sym,shape={
        'data':(4,3,256,256)
    })
    print('############## g2:')
    g2_sym = g2(mx.sym.Variable('data'),  mx.sym.Variable('mask'))
    mx.viz.print_summary(g2_sym,shape={
        'data':(4,3,256,256),
        'mask':(4,1,256,256)
    })
    print('############## d1:')
    d1_sym = d1(mx.sym.Variable('data'), mx.sym.Variable('mask'))
    mx.viz.print_summary(d1_sym,shape={
        'data':(4,3,256,256),
        'mask':(4,1,256,256),
    })
    print('############## d2:')
    d2_sym = d2(mx.sym.Variable('data'), mx.sym.Variable('mask'), mx.sym.Variable('gt'))
    mx.viz.print_summary(d2_sym,shape={
        'data':(4,3,256,256),
        'mask':(4,1,256,256),
        'gt':(4,3,256,256),
    })