#!/usr/bin/env python
# Called by getCode.sh
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

import numpy as np
import sys

s = np.loadtxt(sys.argv[1]).reshape(4,8,8)
s = np.argmax(s,axis=0).flatten()
st = [str(c) for c in s];
print "".join(st)
