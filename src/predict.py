#!/usr/bin/python3
"""Model Predict

Usage: 
    predict.py [options] <img> <models_folder>
    predict.py (-h | --help)

Options:
    -h --help           Show this message
    -s --seg-only       Use only the Segmentation models
    -o <file>           Place de output into <file> [default: ./prediction.nii]
    -v --verbose        Show the results of the prediction
"""
import os
from docopt import docopt
from os import path
import numpy as np
import nibabel as nib

try:
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import tensorflow as tf
except ModuleNotFoundError:
    import tensorflow as tf

import warnings
warnings.filterwarnings('ignore') 


from utils import data, metrics
from models import ataloglou

def predict(args: dict) -> None:

    # CLI ARGUMENTS
    IMG_PATH = path.join(args["<img>"])
    PRED_PATH = path.join(args["-o"])
    MODEL_PATHS = path.join(args["<models_folder>"])

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

    # LOADING MRI
    mri = nib.load(IMG_PATH)
    X = mri.get_fdata()

    # PREDICT MRI
    x1, y1, z1 = np.array(X.shape)//2 - 75
    x2, y2, z2 = np.array(X.shape)//2 + 75

    Y = np.zeros_like(X, dtype="float64")
    Y = Y
    X = (X - X.mean()) / X.std()
    X_in = X[x1:x2, y1:y2, z1:z2]
    Y[x1:x2, y1:y2, z1:z2] = ataloglou.AtaloglouSeg3D(X_in, **seg_models)[..., 0]

    if not args["--seg-only"]:
        ijk = np.array(center(Y)).astype("int") - 50
        for i, (value, shape) in enumerate(zip(ijk, Y.shape)):
            ijk[i] = (0 if value < 0 else value)
            ijk[i] = (shape - 100 if value + 100 > shape else ijk[i])
        i1, j1, k1 = ijk
        i2, j2, k2 = ijk + 100
        y_pred = Y[i1:i2, j1:j2, k1:k2]
        X_in = X[i1:i2, j1:j2, k1:k2]
        Y[i1:i2, j1:j2, k1:k2] = ataloglou.AtaloglouCorr3D(X_in, y_pred, **corr_models)[..., 0]
        Y = np.where(Y > 0.5, 1, 0).astype("int32")

    mri_out = nib.Nifti1Image(Y, mri.affine, mri.header)

    nib.save(mri_out, PRED_PATH)
    print(f"{PRED_PATH} saved")

    if args["--verbose"]:
        from nilearn import plotting as plot
        Z = np.where(Y > 0.5, 1, 0).astype("int32")
        mri_show = nib.Nifti1Image(Z, mri.affine, mri.header)
        plot.plot_roi(mri_show, IMG_PATH)
        plot.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    predict(args)
