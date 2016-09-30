#!/usr/bin/env python
## Python script for converting hdf5 file to npz
## Minimal error-checking
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import numpy as np
from glob import glob
from tempfile import mkdtemp
import os
import sys

def usage():
    print "USAGE: h5npz.py caffemodel.h5 wts.npz"
    raise SystemExit

if len(sys.argv) != 3:
    usage()

if not os.path.isfile(sys.argv[1]):
    usage()

sh=os.path.dirname(os.path.abspath(__file__)) + '/h5x.sh'
tdir=mkdtemp()

os.system(sh + ' ' + sys.argv[1] + ' ' + tdir)

flist=glob(tdir + '/*.dat')
arrs = {}

for f in flist:
    s = f[0:-4]+'.shp'
    dshape = np.loadtxt(s,dtype=np.int).tolist()
    data = np.loadtxt(f,dtype=np.float32)
    data.shape = dshape
    arrs[f[len(tdir):-4]] = data

np.savez(sys.argv[2],**arrs)


os.system('rm -rf ' + tdir)
