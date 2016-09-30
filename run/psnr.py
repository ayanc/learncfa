# PSNR computation
# Copyright (C) 2016 Ayan Chakrabarti <ayanc@ttic.edu>

import numpy as np

class PSNR:

    def __init__(self):
        self.vals = np.zeros((0,),dtype=np.float32)

    # Compute quantiles of PSNR in image and output
    def add(self,im1,im2,psz):
        psnr = np.sum((im1-im2)**2,axis=2)

        for lp in range(psz):
            if psnr.shape[0] % 2 == 0:
                psnr = psnr[::2,:] + psnr[1::2,:]
            else:
                psnr = psnr[:-1:2,:] + psnr[1:-1:2,:]

            if psnr.shape[1] % 2 == 0:
                psnr = psnr[:,::2] + psnr[:,1::2]
            else:
                psnr = psnr[:,:-1:2] + psnr[:,1:-1:2]

        psz = 2**psz
        psnr = 10.*np.log10(np.float32(3*psz*psz)/psnr)
        
        self.vals = np.concatenate((self.vals,psnr.flatten()))
        

    # Print quantiles
    def show(self):
        qs = np.percentile(self.vals,[25,50,75])
        print("PSNR Quantiles:"+ ",".join(["%.2f" % q for q in qs]));
