# Different kinds of sensor layers.
#
# Bayer - Bayer pattern
# CFZ14 - Sparse pattern of Chakrabarti et al., ICCP 2014
# FLCFA - Some fixed arbitrary pattern specified by a code in the parameter string.
#
# LCFA  - Learnable pattern (as described in paper).
#
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import caffe
import numpy as np
from os import getenv

_ochans=4
_istd=0.1
_fac = np.sqrt(399.0)/8e5

# Sense as bayer pattern
class Bayer(caffe.Layer):
    
    def setup(self,bottom,top):
        pass

    def reshape(self,bottom,top):
        top[0].reshape(bottom[0].shape[0],1,24,24)

    def forward(self,bottom,top):
        top[0].data[:,0,0:24:2,0:24:2] = bottom[0].data[:,1,0:24:2,0:24:2]
        top[0].data[:,0,1:24:2,1:24:2] = bottom[0].data[:,1,1:24:2,1:24:2]
        top[0].data[:,0,1:24:2,0:24:2] = bottom[0].data[:,0,1:24:2,0:24:2]
        top[0].data[:,0,0:24:2,1:24:2] = bottom[0].data[:,2,0:24:2,1:24:2]

    def backward(self,top,propagate_down,bottom):
            pass


# Sense as CFZ14 pattern
class CFZ14(caffe.Layer):
    
    def setup(self,bottom,top):
        pass

    def reshape(self,bottom,top):
        top[0].reshape(bottom[0].shape[0],1,24,24)

    def forward(self,bottom,top):
        top[0].data[:,0,:,:] = bottom[0].data[:,3,:,:]
        top[0].data[:,0,0:24:4,0:24:4] = bottom[0].data[:,1,0:24:4,0:24:4]
        top[0].data[:,0,1:24:4,1:24:4] = bottom[0].data[:,1,1:24:4,1:24:4]
        top[0].data[:,0,1:24:4,0:24:4] = bottom[0].data[:,0,1:24:4,0:24:4]
        top[0].data[:,0,0:24:4,1:24:4] = bottom[0].data[:,2,0:24:4,1:24:4]

    def backward(self,top,propagate_down,bottom):
            pass

# Learnable CFA layer
class LCFA(caffe.Layer):
    
    def setup(self,bottom,top):
        self.blobs.add_blob(1,_ochans,8,8)
        self.blobs.add_blob(1)
        self.blobs[0].data[...] = np.float32(np.random.normal(0,_istd,self.blobs[0].data.shape))
        self.blobs[1].data[0] = 0.

    def reshape(self,bottom,top):
        top[0].reshape(bottom[0].shape[0],1,24,24)
        self.pi = np.empty((1,_ochans,8,8),dtype=np.float32)
        
        if self.param_str == 'test':
            self.test = True
        else:
            self.test = False

        self.lcfa = np.float32(1. + (self.blobs[1].data[0]*_fac)**2.)


    def forward(self,bottom,top):
        # Convert blob to probs
        self.pi = self.blobs[0].data.copy()
        
        if self.test:
            self.pi = np.float32(self.pi == np.max(self.pi,axis=1,keepdims=True))
        else:
            self.pi[...] = self.pi - np.max(self.pi,axis=1,keepdims=True)
            np.exp(self.pi*self.lcfa,out=self.pi)
                                 
        self.pi[...] = self.pi / np.sum(self.pi,axis=1,keepdims=True)

        # Apply to input
        top[0].data[...] = np.sum( np.tile(self.pi,(1,1,3,3)) * bottom[0].data,axis=1,keepdims=True)

    def backward(self,top,propagate_down,bottom):
        self.blobs[1].diff[...] = 0.
        if self.test:
            self.blobs[0].diff[...] = 0.
        else:
            dpiE = np.sum(top[0].diff * bottom[0].data,axis=0,keepdims=True)

            # "Un-tile"
            dpi = np.zeros_like(self.pi)
            for i in range(3):
                for j in range(3):
                    dpi += dpiE[:,:,(i*8):((i+1)*8),(j*8):((j+1)*8)]

            dpi = dpi - np.sum(self.pi * dpi,axis=1,keepdims=True)
            self.blobs[0].diff[...] = self.lcfa*dpi*self.pi
            self.blobs[1].data[0] += 1.

# Sense a fixed pattern given  in paramstr
class FLCFA(caffe.Layer):
    
    def setup(self,bottom,top):
        self.code = np.asarray([int(s) for s in self.param_str],dtype=np.int);
        self.code.shape = (8,8);

    def reshape(self,bottom,top):
        top[0].reshape(bottom[0].shape[0],1,24,24)

    def forward(self,bottom,top):
        for i in range(8):
            for j in range(8):
                top[0].data[:,0,i::8,j::8] = bottom[0].data[:,self.code[i,j],i::8,j::8]

    def backward(self,top,propagate_down,bottom):
            pass
