#!/usr/bin/python3
"""Model Train

Usage: 
    train.py [options] <imgs> <masks> (sagital | coronal | axial) 
    train.py [options] <imgs> <masks> <preds> (sagital | coronal | axial)
    train.py (-h | --help)

Options:
    -h --help           Show this message
    -H --history        Plot the history of the model's performance
    --model <path>      Save the trained model [default: ./model.h5].
    --batch <int>       Batch size [default: 8]
    --epochs <int>      Number of epochs [default: 40]
"""

import os
from docopt import docopt
from os import path
import numpy as np
import tensorflow as tf

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
    MODEL_PATH = args["--model"]
    IMG_PATH = path.join(args["<imgs>"])
    LABEL_PATH = path.join(args["<masks>"])
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])

    # LOADING DATA
    if args["<preds>"] != None:
        from utils import corr_data as data
        PRED_PATH = path.join(args["<pred>"])
        PRED_PATHS = np.sort([path.join(PRED_PATH, img) for img in os.listdir(PRED_PATH)])
        ds = data.read_data(IMG_PATHS, LABEL_PATHS, PRED_PATHS)
    else:
        from utils import data
        ds = data.read_data(IMG_PATHS, LABEL_PATHS)

    ds_train = data.mri_preprocess(ds)
    ds2D_slices = data.get_2D_dataset(ds_train, AXIS)
    ds2D_train = ds2D_slices.filter(lambda x, y: tf.reduce_min(y) != tf.reduce_max(y))

    for DS2D_SIZE, _ in enumerate(ds2D_train): pass
    DS2D_SIZE += 1

    ds2D_train = ds2D_train.batch(BATCH_SIZE, drop_remainder=True).shuffle(buffer_size=100)
    ds2D_train = ds2D_train.prefetch(tf.data.AUTOTUNE)

    # LOADING MODEL
    if not args["<preds>"]:
        model = ataloglou.AtaloglouSeg(input_shape=(150, 150, 1))
    else:
        model = ataloglou.AtaloglouCorr(in_shape=(100, 100, 1))

    # TRAINING
    model_hist = model.fit(ds2D_train.repeat(),
                           epochs=EPOCHS,
                           steps_per_epoch=DS2D_SIZE//BATCH_SIZE)

    if args["--history"]:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(model_hist.history["loss"])
        ax[1].plot(model_hist.history["IoU"])
        ax[1].set_ylim([0, 1])
        fig.savefig(f"{MODEL_PATH[:-2]}.png")

    model.save(MODEL_PATH)
