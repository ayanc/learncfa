#!/bin/bash
# Script to get CFA pattern code string from saved caffemodel.h5 file.
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

tname=`mktemp`

h5dump -d /data/sensor/0 $1 | grep \(.*\):| cut -f 2 -d :| sed s/,//g | tr '\n' ' '  > $tname
`dirname $0`/np2code.py $tname
rm $tname

