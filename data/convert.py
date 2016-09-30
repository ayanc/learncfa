#!/usr/bin/env python
#
# procGSHI(indir)
# Process RAW Gehler-Shi data to produce normalized black-level
# corrected 8-bit pngs.
#
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import numpy as np
from skimage.io import imread, imsave
from glob import glob
import sys



def usage():
    print "USAGE: convert.py /path/to/gehler_shi\n"
    print "Call with path to directory where you have extracted"
    print "the 568 16-bit RAW png files from the Gehler Shi database."
    raise SystemExit


if len(sys.argv) != 2:    
    usage()    

flist=glob(sys.argv[1] + '/*.png')
if len(flist) != 568:
    usage()

for i in range(len(flist)):
    onm = "%04d.png" % (i+1)
    print "Converting " + flist[i] + " to " + onm

    im = imread(flist[i],plugin='freeimage')
    im = np.float64(im)
    if i >= 86:
        im = np.maximum(0.,im-129.)
    im = im / np.max(im)

    imsave(onm,np.uint8(np.round(im*255.)))
