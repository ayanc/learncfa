#!/usr/bin/env python
# Main script for testing different sensor and demosaicking methods.
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import sys
import os
import argparse
from time import time
import numpy as np
from skimage.io import imread, imsave

import sensor as sr
import recon as rc
import psnr

PSZ=6 # Log of psize for PSNR (i.e., 64x64 patches)

def popt():
    p = argparse.ArgumentParser(epilog='Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>')

    p.add_argument('--gpu',\
                       help="Use gpu (optionally specifiy device id).",\
                       type=int,nargs='?',\
                       metavar='dev_id',default=-1,\
                       const=0)
    p.add_argument('--wts',\
                       help=".npz file with learnd model weights.",\
                       required=True,metavar="weights.npz")
    p.add_argument('--cfa',\
                       help="CFA pattern to simulate (default=bayer). "+\
                       "Options are bayer, cfz, or lcfa",\
                       choices=['bayer','cfz','lcfa'],\
                       default='bayer', metavar='pattern')
    p.add_argument('--nstd',\
                       help="Noise standard deviation.",\
                       type=float,\
                       default=0)

    p.add_argument('--out',\
                   help='Save reconstructed images (by default,'+\
                   ' only compute PSNR stats). Is automatically'+\
                   ' true if --sinput is specified.',\
                   action='store_true',default=False,dest='isout')

    p.add_argument('--sinput',\
                   help='Input images are CFA images (i.e., they have already been '+\
                   'sampled according to the CFA pattern with noise. By default, '+\
                   'this script expects RGB images, simulates CFA sampling and noise '+\
                   'addition followed by reconstruction, and computes error stats with '+\
                   'respect to ground truth. Use this option to reconstruct already '+\
                   'sampled images. In this case, output images will be saved but PSNR '+\
                   'will not be computed.',\
                   action='store_true',default=False,dest='isinp')

    
    p.add_argument('files',\
                       help="One or more image file names.",\
                       nargs='+',metavar='image', )


    return p.parse_args()

# Parse arguments
args = popt()

# Initialize reconstruction & load weights
if args.gpu >= 0:
    rc.init_gpu(args.gpu)
rc.load_wts(args.wts)

if args.isinp:
    sfx = 'out'
    print "Saving with suffix ", sfx
else:
    # Begin psnr accumulation
    ps = psnr.PSNR()
    if args.isout:
        sfx = args.cfa + "%d" % np.round(args.nstd*10000);
        print "Saving with suffix ", sfx


# Load sensor code from environment if cfa is learned
if args.cfa == 'lcfa':
    code = os.getenv('LCFA_CODE')
    if code == '':
        sys.exit('Need to set environment variable LCFA_CODE.')


prev_shape = (0,0)
print "Testing with noise std ", args.nstd
for f in args.files:

    # Read and truncate image to 8M x 8N
    img = np.float32(imread(f,plugin='freeimage'))/255.
    img = sr.trunc(img)

    if args.isinp:
        simg = img
        if len(simg.shape) > 2:
            sys.exit('If using --sinput, input images should be single channel.')
        if args.cfa == 'bayer':
            simg = simg/3.
    else:
        # Form observation, and cut off 8x8 boundary from GT
        if args.cfa == 'bayer':
            simg = sr.bayer(img,args.nstd)
        if args.cfa == 'cfz':
            simg = sr.cfz(img,args.nstd)
        if args.cfa == 'lcfa':
            simg = sr.lcfa(img,args.nstd,code)

        img = img[8:-8,8:-8,:]

    # Set up precomputes based on image shape (not included in timing)
    if simg.shape != prev_shape:
        rc.dmS(simg.shape)
        prev_shape = simg.shape

    # Process and time
    sys.stdout.write('Processing ' + f)
    stm=time()
    rimg = rc.demosaick(simg)
    stm=time()-stm
    sys.stdout.write(' Time: ' + ("%.2f"%stm) + 'secs\n')

    if args.isout or args.isinp:
        imsave(f[:-3] + sfx + '.png',\
               np.uint16(np.round(rimg*(256*256-1))),\
               plugin='freeimage')

    if not args.isinp:
        ps.add(rimg,img,PSZ)

# Show quantiles        
if not args.isinp:
    ps.show()
