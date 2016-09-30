#!/bin/bash
# Dump contents of h5 caffemodel data into files
# Called by h5npz.py as h5x.sh modefile.h5 dirname
# Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

h5ls -r $1 | grep Dataset | {
    while read line
    do
	echo $line
	dname=`echo $line | cut -f 1 -d ' '`
	shape=`echo $line | grep -o \{.*\} | sed s/[\{\},]//g`
	fname=`echo $dname | cut -f 3- -d '/' | tr '/' '_'`
	echo $shape > $2/$fname.shp
	h5dump -d $dname $1 | grep \(.*\):| cut -f 2 -d :| sed s/,//g | tr '\n' ' ' > $2/$fname.dat
    done
}
	

