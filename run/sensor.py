# Copyright (C) 2016 Ayan Chakrabarti <ayanc@ttic.edu>
import numpy as np

def trunc(img):
    w = img.shape[0]; h = img.shape[1]
    w = (w//8)*8
    h = (h//8)*8
    return img[0:w,0:h,...].copy()
    

def _clip(img):
    return np.maximum(0.,np.minimum(1.,img))

def bayer(img,nstd):
    v = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    v[0::2,0::2] = img[0::2,0::2,1]
    v[1::2,1::2] = img[1::2,1::2,1]
    v[1::2,0::2] = img[1::2,0::2,0]
    v[0::2,1::2] = img[0::2,1::2,2]

    v = v/3.0 + np.float32(np.random.normal(0,1,v.shape))*nstd

    return _clip(v)

def cfz(img,nstd):
    v = np.sum(img,axis=2)
    v[0::4,0::4] = img[0::4,0::4,1]
    v[1::4,1::4] = img[1::4,1::4,1]
    v[1::4,0::4] = img[1::4,0::4,0]
    v[0::4,1::4] = img[0::4,1::4,2]

    v = v/3.0 + np.float32(np.random.normal(0,1,v.shape))*nstd

    return _clip(v)

def lcfa(img,nstd,code_str):
    code = np.asarray([int(s) for s in code_str],dtype=np.int);
    code.shape = (8,8);

    v = np.sum(img,axis=2)
    for i in range(8):
        for j in range(8):
            if code[i,j] < 3:
                v[i::8,j::8] = img[i::8,j::8,code[i,j]]

    v = v/3.0 + np.float32(np.random.normal(0,1,v.shape))*nstd
    return _clip(v)
    
