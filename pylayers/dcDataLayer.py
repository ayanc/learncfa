# Data layer to load data.
#
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import caffe
import numpy as np

from string import split
from os import getenv
from skimage.io import imread

_ochans=4

_cspace = np.zeros((4,3),dtype=np.float32)
_cspace[0:3,0:3] = np.eye(3,dtype=np.float32)
_cspace[3,:] = np.ones((1,3),dtype=np.float32)
_cspace = _cspace/3.

class DataRGBW(caffe.Layer):
    
    def setup(self,bottom,top):
        sdir = getenv('DC_DATA_DIR')
        if sdir is None:
            raise Exception("Set environment variable DC_DATA_DIR "
                            " to data directory.")

        params = split(self.param_str,':')
        if len(params) != 5:
            raise Exception("DataRGBW params must be of the form "
                            "listfile:batch_size:chunk_size:"
                            "chunk_repeat:noise_std")

        # Read list of files
        self.flist = [sdir + "/" + line.rstrip('\n') for
                      line in open(sdir + "/" + params[0])]

        # Read remaining params
        self.bsize, self.csize = int(params[1]), int(params[2])
        self.crep, self.nstd = int(params[3]), float(params[4])

        self.test = False
        if self.csize == 0:
            self.csize = 1;
            self.test = True


        # Shuffle file list
        if not self.test:
            np.random.shuffle(self.flist)

        # Setup walk through database
        self.imid, self.chunkid = 0, 0

    def reshape(self,bottom,top):
        top[0].reshape(self.bsize,_ochans,24,24)
        top[1].reshape(self.bsize,3,8,8)

    def forward(self,bottom,top):
        # Read a new chunk if necessary
        if self.chunkid == 0:
            self.imgs = []
            for i in range(self.csize):
                self.imgs.append(np.float32(imread(self.flist[self.imid]))/255.)
                self.imid = (self.imid+1) % len(self.flist)
        
            if self.test: # Sample uniformly
                shp = self.imgs[0].shape
                nump = (shp[0]-24)*(shp[1]-24)
                pidx = [int(i) for i in np.linspace(0,nump-1,self.crep*self.bsize)]
                self.xs = [i%(shp[0]-24) for i in pidx]
                self.ys = [i/(shp[0]-24) for i in pidx]
                self.pidx = 0

        self.chunkid = (self.chunkid+1)%self.crep

        for i in range(self.bsize):
            # Find random image and location
            if self.test:
                iid = 0; x = self.xs[self.pidx]; y = self.ys[self.pidx]
                self.pidx += 1
            else:
                iid = np.random.randint(self.csize)
                x = np.random.randint(self.imgs[iid].shape[0]-24)
                y = np.random.randint(self.imgs[iid].shape[1]-24)

            # Permute array dims
            im = self.imgs[iid][x:(x+24),y:(y+24),:].transpose(2,0,1).copy()

            # Crop center patch as gt output
            top[1].data[i,...] = im[:,8:16,8:16].reshape((1,3,8,8))
            
            # Creat input to sensor, add noise
            im.shape = (3,24*24); im = np.dot(_cspace,im)
            im[...] = im + np.random.normal(0,self.nstd,im.shape)
            np.maximum(0.,im,im); np.minimum(1.,im,im);
            top[0].data[i,...] = im.reshape(1,_ochans,24,24)

    def backward(self,top,propagate_down,bottom):
            pass
