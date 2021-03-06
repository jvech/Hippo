#!/usr/bin/python3
"""Model Eval

Usage: 
    eval.py [options] <imgs> <masks> <models>
    eval.py (-h | --help)

Options:
    -h --help           Show this message
    -s --seg-only       Use only the Segmentation models
"""
import os
from docopt import docopt
from os import path
import numpy as np

try:
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import tensorflow as tf
except ModuleNotFoundError:
    import tensorflow as tf



from utils import data, metrics
from models import ataloglou

def model_eval(args: dict) -> None:

    # CLI ARGUMENTS
    IMG_PATH = path.join(args["<imgs>"])
    LABEL_PATH = path.join(args["<masks>"])
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])
    MODEL_PATHS = path.join(args["<models>"])

    #LOAD MODEL
    for File in os.listdir(MODEL_PATHS):
        if "seg" in File and "sagit" in File and File.endswith("h5"):
            SAGIT_PATH = path.join(MODEL_PATHS, File)
            sagit_model = tf.keras.models.load_model(SAGIT_PATH)
        elif "seg" in File and "coron" in File and File.endswith("h5"):
            CORON_PATH = path.join(MODEL_PATHS, File)
            coron_model = tf.keras.models.load_model(CORON_PATH)
        elif "seg" in File and "axial" in File and File.endswith("h5"):
            AXIAL_PATH = path.join(MODEL_PATHS, File)
            axial_model = tf.keras.models.load_model(AXIAL_PATH)

        if not args["--seg-only"]:
            if "corr" in File and "sagit" in File and File.endswith("h5"):
                SAGIT_PATH_CORR = path.join(MODEL_PATHS, File)
                sagit_model_corr = tf.keras.models.load_model(SAGIT_PATH_CORR)
            elif "corr" in File and "coron" in File and File.endswith("h5"):
                CORON_PATH_CORR = path.join(MODEL_PATHS, File)
                coron_model_corr = tf.keras.models.load_model(CORON_PATH_CORR)
            elif "corr" in File and "axial" in File and File.endswith("h5"):
                AXIAL_PATH_CORR = path.join(MODEL_PATHS, File)
                axial_model_corr = tf.keras.models.load_model(AXIAL_PATH_CORR)

    # LOAD MODEL 
    sagit_model = tf.keras.models.load_model(SAGIT_PATH)
    coron_model = tf.keras.models.load_model(CORON_PATH)
    axial_model = tf.keras.models.load_model(AXIAL_PATH)

    seg_models = {'sagit': sagit_model, 
                  'coron': coron_model, 
                  'axial': axial_model}

    if not args["--seg-only"]:
        from utils.corr_data import center
        corr_models = {'sagit': sagit_model_corr, 
                       'coron': coron_model_corr, 
                       'axial': axial_model_corr}

    # LOADING DATASET
    ds = data.read_data(IMG_PATHS, LABEL_PATHS)
    ds_train = data.mri_preprocess(ds)

    scores = []
    for X, Y in ds_train:
        y_pred = ataloglou.AtaloglouSeg3D(X, **seg_models)[..., 0]
        y = Y
        if not args["--seg-only"]:
            ijk = np.array(center(y_pred)).astype("int") - 50
            for i, (value, shape) in enumerate(zip(ijk, y_pred.shape)):
                ijk[i] = (0 if value < 0 else value)
                ijk[i] = (shape - 100 if value + 100 > shape else ijk[i])
            i, j, k = ijk
            y_pred = y_pred[i:i+100, j:j+100, k:k+100]
            x = X[i:i+100, j:j+100, k:k+100]
            y = Y[i:i+100, j:j+100, k:k+100]
            y_pred = ataloglou.AtaloglouCorr3D(x, y_pred, **corr_models)[..., 0]
        y_pred = tf.where(y_pred > 0.5, 1, 0)
        scores.append(metrics.get_scores(y, y_pred))

    scores = np.array(scores).mean(axis=0), np.array(scores).std(axis=0)
    metrics.print_scores("Ataloglou performance", scores)


if __name__ == "__main__":
    args = docopt(__doc__)
    model_eval(args)
