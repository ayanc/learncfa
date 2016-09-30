# Reconstruction pipeline implementation in Python for CPU and GPU
# Applies the caffe network using numpy / pycuda routines.
#
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import numpy as np
from time import time

# Flag for GPU usage
_gpu=False

# Values loaded from weight file
class _WTS:
    pass
_wts = _WTS()
_eps=0.00000001
_ofac=0
_numsel=0

# Initialize GPU
def init_gpu(dev=0):
    global gp, lg, cm, msc, slf, _gpu
    
    from pycuda import gpuarray as gp
    from pycuda import elementwise as ew
    from pycuda import cumath as cm
    from skcuda import linalg as lg
    from skcuda import misc as msc

    msc.init_context(msc.init_device(dev))
    lg.init()

    slf = ew.ElementwiseKernel(
        "float * y, float * x, unsigned * ind",
        "y[i] = x[ind[i]]"
    )

    _gpu = True

# Load and permute variables from a npz file
def load_wts(file):
    global _wts, _eps, _ofac, _numsel

    wf = np.load(file)

    ## Interpolation path
    # Transpose output of IP(log) layer so that channels is the fastest dimension
    int1 = wf['int1_0']
    _ofac = int1.shape[0]/64/3
    int1.shape = (_ofac*3,64,576)
    _wts.int1 = int1.transpose([2,1,0]).reshape(576,_ofac*3*64).copy()

    # Reshape copies of 3-chans so that we can do tail-reduction
    int2 = wf['intOut_0']; int2b = wf['intOut_1']
    int2.shape = (_ofac,3,_ofac*3); int2b.shape = (_ofac,3)
    _wts.int2 = int2.transpose([2,1,0]).reshape(_ofac*3,_ofac*3).copy()
    _wts.int2b = int2b.transpose([1,0]).reshape(1,_ofac*3).copy()

    ## Selection path
    c1 = wf['conv1_0']; c1b = wf['conv1_1']
    _numsel = c1.shape[0]
    _wts.c1 = c1.transpose([2,3,1,0]).reshape(64,_numsel).copy()
    _wts.c1b = c1b.reshape(1,_numsel).copy()

    c2 = wf['conv2_0']; c2b = wf['conv2_1']
    _wts.c2b = c2b.reshape(1,_numsel).copy()
    _wts.c200 = c2[:,:,0,0].transpose([1,0]).copy()
    _wts.c201 = c2[:,:,0,1].transpose([1,0]).copy()
    _wts.c210 = c2[:,:,1,0].transpose([1,0]).copy()
    _wts.c211 = c2[:,:,1,1].transpose([1,0]).copy()

    c3 = wf['conv3_0']; c3b = wf['conv3_1']
    _wts.c3b = c3b.reshape(1,_numsel).copy()
    _wts.c300 = c3[:,:,0,0].transpose([1,0]).copy()
    _wts.c301 = c3[:,:,0,1].transpose([1,0]).copy()
    _wts.c310 = c3[:,:,1,0].transpose([1,0]).copy()
    _wts.c311 = c3[:,:,1,1].transpose([1,0]).copy()

    sout = wf['sOut0_0']; soutb = wf['sOut0_1']
    sout.shape = (_ofac,3,64,_numsel)
    soutb.shape = (_ofac,3,64)
    _wts.sout = sout.transpose([3,2,1,0]).reshape(_numsel,_ofac*3*64).copy()
    _wts.soutb = soutb.transpose([2,1,0]).reshape(1,_ofac*3*64).copy()

    # Move everything to a gpu if necessary
    if _gpu:
        for p in dir(_wts):
            a = getattr(_wts,p)
            if type(a).__module__ == 'numpy':
                setattr(_wts,p,gp.to_gpu(a))


# Set up index calculations for im2col
def im2col_I(ishape,block,stride):
    bidx = np.array(range(block),dtype=np.uint32).reshape(block,1)*ishape[1]
    bidx = bidx + np.array(range(block),dtype=np.uint32)
    bidx.shape = (1,1,block*block)


    idx = np.array(range(ishape[0]*ishape[1]),dtype=np.uint32)
    idx.shape = ishape
    idx = idx[0:1-block:stride,0:1-block:stride].copy()
    idx.shape = (idx.shape[0],idx.shape[1],1)
    
    idx = idx + bidx;oshape = idx.shape
    idx.shape = (oshape[0]*oshape[1]*oshape[2],)
    
    if _gpu:
        idx = gp.to_gpu(idx)

    i2c = [(ishape[0]*ishape[1]), idx, oshape]
    return i2c


# Do im2col with pre-computed indices
def im2col(img,i2c):
    if _gpu:
        img = img.reshape(i2c[0])
        out = gp.empty(len(i2c[1]),dtype=np.float32)
        slf(out,img,i2c[1],range=slice(0, len(i2c[1]), 1))
        out = out.reshape(i2c[2])
    else:
        out = img.reshape(i2c[0])[i2c[1]]
        out.shape = i2c[2]

    return out
    
# Re-organize patches into image
def p2im(img):
    w = img.shape[0]; h = img.shape[1]
    out = np.zeros((w*8,h,8*3),dtype=np.float32)
    for i in range(8):
        out[i::8,...] = img[:,:,(i*24):((i+1)*24)]

    out.shape = [w*8,h*8,3]
    return out


# Set up image-size specific calculations
def dmS(ishape):
    global _i2c1, _i2c2
    _i2c1 = im2col_I(ishape,24,8)
    _i2c2 = im2col_I(ishape,8,8)


# Reconstruction implementation on GPU
def demosaick_gpu(img):
    img = gp.to_gpu(img)
    p2x = im2col(img,_i2c2)
    cm.log(img+_eps,out=img)
    p1x = im2col(img,_i2c1)

    wA = p1x.shape[0]; wB = p2x.shape[0]
    hA = p1x.shape[1]; hB = p2x.shape[1]

    # Path 1
    p1x = p1x.reshape([wA*hA,576])
    p1y = lg.dot(p1x,_wts.int1)
    cm.exp(p1y,out=p1y)

    p1y = p1y.reshape([wA*hA*64,3*_ofac])
    p1x = lg.dot(p1y,_wts.int2) 
    msc.add_matvec(p1x,_wts.int2b,out=p1x)
    p1x = p1x.reshape([wA*hA*64*3,_ofac])

    # Path 2
    # conv1
    p2x = p2x.reshape([wB*hB,64])
    p2y = lg.dot(p2x,_wts.c1) 
    msc.add_matvec(p2y,_wts.c1b,out=p2y)
    gp.maximum(p2y,0.,p2y)
    p2y = p2y.reshape([wB,hB,_numsel])

    # conv2
    shI = [wB-1,hB-1,_numsel]; shM = [(wB-1)*(hB-1),_numsel]
    p2x = gp.empty(shM,dtype=np.float32)
    pTT = gp.empty(shI,dtype=np.float32)

    pTT = pTT.reshape(shI); pTT[...] = p2y[0:-1,0:-1,:]; pTT = pTT.reshape(shM);
    p2x = lg.dot(pTT,_wts.c200)
    pTT = pTT.reshape(shI); pTT[...] = p2y[0:-1,1:,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c201,p2x)
    pTT = pTT.reshape(shI); pTT[...] = p2y[1:,0:-1,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c210,p2x)
    pTT = pTT.reshape(shI); pTT[...] = p2y[1:,1:,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c211,p2x)
    msc.add_matvec(p2x,_wts.c2b,out=p2x)
    gp.maximum(p2x,0.,p2x)
    p2x = p2x.reshape(shI)

    # conv 3
    shI = [wB-2,hB-2,_numsel]; shM = [(wB-2)*(hB-2),_numsel]
    p2y = gp.empty(shM,dtype=np.float32)
    pTT = gp.empty(shI,dtype=np.float32)

    pTT = pTT.reshape(shI); pTT[...] = p2x[0:-1,0:-1,:]; pTT = pTT.reshape(shM);
    p2y = lg.dot(pTT,_wts.c300)
    pTT = pTT.reshape(shI); pTT[...] = p2x[0:-1,1:,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c301,p2y)
    pTT = pTT.reshape(shI); pTT[...] = p2x[1:,0:-1,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c310,p2y)
    pTT = pTT.reshape(shI); pTT[...] = p2x[1:,1:,:]; pTT = pTT.reshape(shM);
    lg.add_dot(pTT,_wts.c311,p2y)
    msc.add_matvec(p2y,_wts.c3b,out=p2y)
    gp.maximum(p2y,0.,p2y)

    p2x = lg.dot(p2y,_wts.sout)

    msc.add_matvec(p2x,_wts.soutb,out=p2x)
    gp.maximum(p2x,0.,p2x)
    p2x = p2x.reshape(p1x.shape)

    # Combine
    p1x *= p2x
    p1 = msc.sum(p1x,axis=1)
    gp.maximum(p1,0.,p1)
    gp.minimum(p1,1.,p1)
    p1 = p1.reshape([wA,hA,64*3])

    im = p2im(p1.get())

    return im

# Reconstruction implementation on CPU
def demosaick_cpu(img):

    p1 = im2col(np.log(img+_eps),_i2c1)
    p2 = im2col(img,_i2c2)

    wA = p1.shape[0]; wB = p2.shape[0]
    hA = p1.shape[1]; hB = p2.shape[1]


    # Path 1
    p1.shape = [wA*hA,576]
    p1 = np.dot(p1,_wts.int1)
    p1 = np.exp(p1)
    p1.shape = [wA*hA*64,3*_ofac]
    p1 = np.dot(p1,_wts.int2) + _wts.int2b
    p1.shape = [wA,hA,64,3,_ofac]

    # Path 2
    p2.shape = [wB*hB,64]
    p2 = np.dot(p2,_wts.c1) + _wts.c1b
    np.maximum(p2,0.,p2)

    p2.shape = [wB,hB,_numsel]
    p00 = p2[0:-1,0:-1,:].copy(); p00.shape = [(wB-1)*(hB-1),_numsel]
    p01 = p2[0:-1,1:,:].copy(); p01.shape = p00.shape
    p10 = p2[1:,0:-1,:].copy(); p10.shape = p00.shape
    p11 = p2[1:,1:,:].copy(); p11.shape = p00.shape

    p2 = _wts.c2b + np.dot(p00,_wts.c200) + np.dot(p01,_wts.c201) + np.dot(p10,_wts.c210) + np.dot(p11,_wts.c211)
    np.maximum(p2,0.,p2)
    
    p2.shape = [wB-1,hB-1,_numsel]
    p00 = p2[0:-1,0:-1,:].copy(); p00.shape = [(wB-2)*(hB-2),_numsel]
    p01 = p2[0:-1,1:,:].copy(); p01.shape = p00.shape
    p10 = p2[1:,0:-1,:].copy(); p10.shape = p00.shape
    p11 = p2[1:,1:,:].copy(); p11.shape = p00.shape

    p2 = _wts.c3b + np.dot(p00,_wts.c300) + np.dot(p01,_wts.c301) + np.dot(p10,_wts.c310) + np.dot(p11,_wts.c311)
    np.maximum(p2,0.,p2)

    p2 = _wts.soutb + np.dot(p2,_wts.sout)
    np.maximum(p2,0.,p2)

    p2.shape = p1.shape

    # Combine
    p1 = np.sum(p1*p2,axis=4)
    p1.shape = [wA,hA,64*3]

    np.maximum(p1,0.,p1)
    np.minimum(p1,1.,p1)

    return p2im(p1)


# Main demosaicking function entry point
def demosaick(img):
    if _gpu:
        return demosaick_gpu(img)
    else:
        return demosaick_cpu(img)
