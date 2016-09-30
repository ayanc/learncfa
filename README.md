## Learning Sensor Multiplexing Design through Back-propagation
Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

This distribution provides a framework for *learning* the color
multiplexing pattern of a digital color camera, jointly with a neural
network to interpolate and reconstruct full color images from the
corresponding sensor measurements. It is a reference implementation of
the algorithm described in the paper:

Ayan Chakrabarti, "**[Learning Sensor Multiplexing Design through 
Back-propagation][paper],**" Advances in Neural Information Processing 
Systems (NIPS) 2016.

This implementation is being made available for non-commercial
research use only. If you find this code useful in your research,
please cite the above paper.  See the [project page] for further
details, and contact <ayanc@ttic.edu> with questions.

**Table of Contents**

1. [Requirements](#requirements)
2. [Preparing Data](#preparing-data)
3. [Pre-trained Models](#pre-trained-models)
4. [Reconstruction with Trained Models](#reconstruction-with-trained-models)
5. [Training Sensor Patterns and Reconstruction Networks](#training-sensor-patterns-and-reconstruction-networks)

## Requirements

We use [caffe] for training, using custom Python layers to simulate
the input incident light and our learnable sensor layer. Therefore, if
you want to train your own models (rather than just use the ones
provided), you will need to compile **caffe** with support for Python
layers. Moreover, you will need the [skimage] package for Python
installed on your system. Make sure that it is installed with the
`freeimage` plugin for reading and saving files. You will also need
the command line [HDF5 Tools] to run the scripts that convert hdf5
caffe models to a python readable format.


We use a purely Python-based script, without any caffe dependencies,
for reconstruction using a trained neural network model. These scripts
require **skimage** to be installed. Moreover, if you want to these
scripts to take advantage of a GPU, you will need to install the
following python packages: [pycuda] and [scikit-cuda].

## Preparing Data

We use the [Gehler-Shi] dataset for training, validation, and
testing. You will need to download the dataset and process them using
the script `convert.py` provided in the `data/` directory. To do so,
download the image archives from the dataset page, and extract them
into some location. Then, from within the `data/` directory of this
distribution, run:

```bash
$ cd data
$ ./convert.py /path/to/gehler_shi
```

where `/path/to/gehler_shi` is the directory where you extracted the
database images. This will convert the original RAW images into
normalized 8-bit PNG files, applying the required black-level
correction to the Canon 5D images.  The `data/` directory already
contains text files listing the split of this database into the
training, validation, and test sets used in the experiments in the
paper.

## Pre-trained Models

You can download pre-trained reconstruction network models for
different patterns and different noise levels as a `.zip` archive from
the [project page]. These models are in Python's `.npz` format, with
file names indicating the sensor pattern and noise level.

Specifically, `bayer*.npz` are for reconstructing from images sampled
with the Bayer pattern, `cfz*.npz` for the sparse pattern of
[Chakrabarti et al. 2014], and `lcfa*.npz` for our learned
pattern. This learned pattern corresponds to the following "*code
string*" (see the next section for an explanation, and how to provide
the code string to the reconstruction script):

```
3133033303333303132232330233131233333103303133100323133331333322
```

The second part of each model's filename indicates the noise level for
which it was trained. `*0.npz` are trained for noiseless observations,
`*25.npz` for AWGN with standard deviation 0.0025, `*50.npz` for
std. 0.005, and so on.  These standard deviations are with respect to
the input color image intensities in the range `[0,1]`.


## Reconstruction with Trained Models

The script `run/runDM.py` can be used to evaluate different CFA
patterns and reconstruction networks. You can use these scripts with
the provided pre-trained models, or for the models that you train
yourself. Call the script with `-h` to get the full list of
options. We given an overview of its two modes of operation below.

**Simulate Sampling and Reconstruction**: This is the standard mode of
operation. Call the script on a set of full color RGB images: for each
image, the script will simulate sampling with a specified sampling
pattern with noise at a specified standard deviation, followed by
reconstruction with the provided model. The script supports sampling
with the Bayer pattern, [[Chakrabarti et al. 2014]]'s pattern, and an
arbitrary pattern specified by a *code string*-to handle learned
patterns. By default, the script only computes PSNR statistics in this
setting, but you can specify an option to save the reconstructed
images as well (as 16-bit PNGs).

Arbitrary patterns are specified by setting the environment variable
**`LCFA_CODE`** to a *code string*, which is a sequence of 64 digits,
each between 0 and 3, that specify which color is sampled at each
pixel in the 8x8 pattern (serialized in reading order), with 0
indicating Red, 1 Green, 2 Blue, and 3 White.

For example, to compute reconstruction PSNR statistics on the learned
sampling pattern with the provided pre-trained models, at noise
std. 0.005, call the script as follows:

```bash
$ export LCFA_CODE=3133033303333303132232330233131233333103303133100323133331333322
$ cd data/
$ ../run/runDM.py --cfa lcfa --nstd 0.005 --wts /path/to/models/lcfa50.npz  `cat test.txt`
```

**Reconstruct Already Sampled Images**: You can also call the script
on already sampled single channel images, in which case it will apply
the reconstruction network on them directly. In this setting, the
script outputs the reconstructed image, but does not compute PSNR
stats (since it doesn't have access to the ground truth). To generate
your own sampled images:

- Please ensure that you use the same pattern layout convention as our
  code (see `data/sensor.py` or `pylayers/sensor.py` for
  reference). In particular, in Bayer-sampled images, the top left
  (x=0,y=0) pixel should be green, with (x=1,y=1) also green,
  (x=0,y=1) red, and (x=1,y=0) blue.

- Since *white* intensities are assumed to be the sum of red, green,
  and blue, to ensure that all sampled intensities lie in the range
  [0,1], we divide the original RGB intensities by a factor of 3. The
  noise standard deviation is with respect to this scaling of the
  intensities, and the reconstruction network models are trained to go
  from this input and output RGB values at their original scale.

  You should follow the same intensity-scaling procedure if you
  generate your own sampled images. The only exception is for the
  Bayer pattern: as a convenience (and to enable comparisons to other
  Bayer demosaicking algorithms), the script assumes that intensities
  in provided Bayer-sampled images have NOT been divided by 3.
  Therefore, the script will scale down the intenities in the input
  images before passing them to the reconstruction network.

We recommend you read through the source code for a clearer
understanding of these conventions.


## Training Sensor Patterns and Reconstruction Networks

All training is done using [caffe]. The directory `prototxt/` contains
network definition proto-texts for different kinds of training: (1)
training the sensor pattern jointly with a reconstruction network, and
(2) training a reconstruction network for a fixed sensor pattern. The
latter case is used both for the pattern we learn from (1), as well as
for training competing reconstruction networks for traditional sensing
patterns.

### Custom Layers

These networks use various custom layers defined in python, and
provided in the `pylayers/` directory. You will need to ensure that
this directory, along with the `caffe/python` directory, are included
in your `PYTHONPATH` environment variable before calling the `caffe`
executable. We provide a brief description of these layers below, and
you can look at the proto-text files to see them in action.

**Data Layer**: The `dcDataLayer.py` file provides the `DataRGBW`
layer. This layer loads 24x24 image patches cropped from images in a
dataset (like the [Gehler-Shi] dataset that we use), and produces two
"tops" for each patch: (1) a 4-channel blob (corresponding to noisy
red, green, blue and white intensity measurements) which will be
sub-sampled by a *sensor* layer, and (2) the ground truth RGB
intensities of the center 8x8 patch that the network must learn to
reconstruct.

The layer takes a parameter string (see proto-text files) in the
following format:
`list.txt:batch_size:chunk_size:chunk_repeat:noise_std`

Here, `list.txt` is the (full path to) a text file containing the list
of images to load patches from: you can use the `train.txt` and
`val.txt` files in the data directory of this distribution for
training and validation, to recreate our experimental setup. You will
also need to set the environment variable `DC_DATA_DIR` to the full
path of the directory that contains these image files.

`noise_std` is simply the standard deviation of the AWGN noise to be
added to the measurements of the first blob. `batch_size` is the
number of patches to include in each batch, while `chunk_size` and
`chunk_repeat` define the patch-sampling behavior. Specifically, the
layer will load a set of `chunk_size` images every `chunk_repeat`
iterations, and in each iteration, sample patches randomly from this
set of images. This ensures that we minimize disk access and use
multiple patches from the same image, but that every batch is diverse
and has samples from multiple images.

Of course, for validation, you don't want this sampling to be random,
since you want to be able to compare loss on the same patches every
time you validate. If `chunk_size` is set to 0 (as it is in the `TEST`
phase of your proto-text), then the data layer will go through the
list of images one image at a time, using each image to generate
batches for `chunk_repeat` iterations.

**Sensor Layers**: The `sensor.py` file provides four layer classes,
each of which take in the channel 4-channel blob produced by the data
layer, and output a single channel blob formed by sampling a specific
channel at each location. Each class uses a different sampling
pattern: `Bayer` samples according to the Bayer pattern, `CFZ14`
according to [[Chakrabarti et al. 2014]]'s pattern, and `FLCFA`
according to an arbitrary pattern specified as a code string in its
parameter string. The fourth kind of sensor layer is `LCFA` that
actually learns the sensor pattern. A soft encoding of the pattern
(see the [paper] for details) is a learnable parameter blob for this
layer, and will be updated by caffe using SGD, and stored with the
reconstruction weights in saved `caffemodel` files.

### Training Procedure

The network architectures for various training settings are provided
in the files in the `prototxt/` directory. Note that these are only
the network proto-texts. You will need to create a `solver.prototxt`
file that: (a) saves weights in the HDF5 format (instead of protobuf),
(b) trains using SGD with a momentum of 0.9, and (c) follows a
learning rate schedule from the paper (and also detailed below). If
you want to test on the validation set, you should set `test_iter` to
the number of validation images (51 in our split defined in
`data/val.txt`).

In the experiments in the paper, we first learned the sensor pattern
at a noise-level of 0.01. The file `learncfa.prototxt` provides the
network architecture to achieve this (if you want to learn the pattern
at a different noise-level, you will have to modify the last parameter
in the `param_str` for the data layers). To recreate the setting in
the paper, train this network for `1.5e6` iterations with a learning
rate of `0.001`. Once you are done training, call the script
`h5proc/getCode.sh` with a path to the trained caffemodel.h5 file, to
retrieve the sensor *code string*.

We then train reconstruction networks at several different noise
levels using this code, and the `FLCFA` sensor layer, passing the
*code string* retrieved in the previous step in the `param_str` of
that layer. The file `flcfa50.prototxt` provides the description of
the network architecture for this (for a noise level of
`0.005`). These networks are trained for `1.5e6` iterations with a
learning rate of `0.001`, and another `1e5` iterations with
`lr=0.0001`. 

The `bayer50.prototxt` and `cfz50.prototxt` files can be used for
training reconstruction networks for the Bayer and
[[Chakrabarti et al. 2014]] pattern respectively. These should be
trained with the same schedule as the `flcfa50.prototxt` network:
`1.5e6` iterations with `lr=0.001`, followed by `1e5` iterations with
`lr=0.0001`.

After you are done training the reconstruction networks, you can use
the `h5proc/h5npz.py` script, e.g.:
```shell
h5npz.py /path/to/wts/bayer25_iter_1500000.caffemodel.h5 bayer25.npz
```
to convert the stored caffe model into the `.npz` format used by our
reconstruction script `run/runDM.py`. Note that the architecture of
the network is hard-coded into the reconstruction script, so if you
choose to use a different reconstruction architecture, you'll have to
make corresponding changes in the `runDM.py` script as well.

[paper]: https://arxiv.org/abs/1605.07078
[project page]: http://www.ttic.edu/chakrabarti/learncfa/
[caffe]: http://www.github.com/BVLC/caffe
[skimage]: http://scikit-image.org/
[HDF5 Tools]: https://support.hdfgroup.org/HDF5/doc/RM/Tools.html
[pycuda]: https://mathema.tician.de/software/pycuda/
[scikit-cuda]: https://github.com/lebedov/scikit-cuda/
[Gehler-Shi]: http://www.cs.sfu.ca/~colour/data/shi_gehler/
[Chakrabarti et al. 2014]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6831801
