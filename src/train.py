#!/usr/bin/python3
"""Model Train

Usage: 
    train.py [options] <imgs> <masks> (sagital | coronal | axial) 
    train.py (-h | --help)

Options:
    -h --help           Show this message
    -H --history        Plot train history
    --model <path>      Save the trained model [default: ./model.h5].
    --batch <int>       Batch size [default: 8]
    --epochs <int>      Number of epochs [default: 40]
"""

import os
from docopt import docopt
from os import path
import numpy as np
import tensorflow as tf

from utils import data
from models import ataloglou

if __name__ == "__main__":
    args = docopt(__doc__) # I Love You

    # CLI arguments
    if args["sagital"]: AXIS = 0
    elif args["coronal"]: AXIS = 1
    elif args["axial"]: AXIS = 2
    else: AXIS = None

    EPOCHS = int(args["--epochs"])
    BATCH_SIZE = int(args["--batch"])
    IMG_PATH = path.join(args["<imgs>"])
    LABEL_PATH = path.join(args["<masks>"])
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])

    # LOADING DATASET
    ds = data.read_data(IMG_PATHS, LABEL_PATHS)
    ds_train = data.mri_preprocess(ds)
    ds2D_slices = data.get_2D_dataset(ds_train, AXIS)
    ds2D_train = ds2D_slices.filter(lambda x, y: tf.reduce_min(y) != tf.reduce_max(y))

    for DS2D_SIZE, (x, y) in enumerate(ds2D_slices): pass

    ds2D_train = ds2D_train.batch(BATCH_SIZE, drop_remainder=True).shuffle(buffer_size=1000)
    ds2D_train = ds2D_train.prefetch(tf.data.AUTOTUNE)

    # LOADING MODEL
    model = ataloglou.AtaloglouSeg()

    # TRAINING
    model_hist = model.fit(ds2D_train.repeat(),
                           epochs=EPOCHS,
                           steps_per_epoch=DS2D_SIZE//BATCH_SIZE)

