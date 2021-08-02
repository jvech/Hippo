import tensorflow as tf
import nibabel as nib
import numpy as np
import os

from os import path


def read_data(img_paths, pred_paths, label_paths):
    def mri_read():
        for X, Z, Y in zip(img_paths, pred_paths, label_paths):
            yield ((tf.constant(nib.load(X).get_fdata(), "float64"),
                   tf.constant(nib.load(Z).get_fdata(), "float64")),
                   tf.constant(nib.load(Y).get_fdata(), "float64"))

    DS_DIMS = nib.load(img_paths[0]).shape
    ds = tf.data.Dataset.from_generator(
                mri_read,
                output_signature = ((tf.TensorSpec(DS_DIMS, tf.float64), 
                                    tf.TensorSpec(DS_DIMS, tf.float64)),
                                    tf.TensorSpec(DS_DIMS, tf.float64))
                )

    return ds

def mri_preprocess(ds):
    def preprocessing3D(d, y):
        ## X = (X - X_mean)/X_std
        x, z = d
        x = (x - tf.reduce_mean(x))/ tf.math.reduce_std(x)
        y = tf.where(y!=0, 1, 0)
        return (x, z), y

    def extract_roi(d, y, ijk=(40, 50, 20), whl=(100, 100, 100)):
        x, z = d
        i, j, k = ijk
        w, h, l = whl
        return (x[i:i+w, j:j+h, k:k+l], z[i:i+w, j:j+h, k:k+l]), y[i:i+w, j:j+h, k:k+l]

    def center(X):
        c_x = np.sum(np.sum(X, axis=(1, 2)) * np.arange(X.shape[0])) / np.sum(X)
        c_y = np.sum(np.sum(X, axis=(0, 2)) * np.arange(X.shape[1])) / np.sum(X)
        c_z = np.sum(np.sum(X, axis=(0, 1)) * np.arange(X.shape[2])) / np.sum(X)
        return int(c_x), int(c_y), int(c_z)

    pre_ds = ds.map(lambda x, y: preprocessing3D(x, y))
    centers = np.array([center(y) for (x, z), y in pre_ds]).mean(axis=0).astype("int")
    ijk = tuple(centers.astype("int") - 50)
    return pre_ds.map(lambda x, y: extract_roi(x, y, ijk=ijk))

def get_2D_dataset(ds, axis):
    def get_axis_view(D, Y):
        X, Z = D
        d = []
        y = []
        for i in range(X.shape[axis]):
            d.append((X[(slice(None),)*axis + (i,)], Z[(slice(None),)*axis + (i,)]))
            y.append(Y[(slice(None),)*axis + (i,)])
        d = tf.data.Dataset.from_tensor_slices(d)
        y = tf.data.Dataset.from_tensor_slices(y)
        return tf.data.Dataset.zip((d, y))

    def mri_2D_preprocess(d, y):
        x, z = d[0], d[1]
        x = tf.expand_dims(x, -1)
        z = tf.expand_dims(z, -1)
        return (x, z), tf.expand_dims(y, -1)

    ds_2D = ds.flat_map(lambda x, y: get_axis_view(x, y))
    ds_2D = ds_2D.map(lambda x, y: mri_2D_preprocess(x, y))

    return ds_2D

if __name__ == "__main__":
    IMG_PATH = path.join("../input/images_35")
    LABEL_PATH = path.join("../input/Labels_35")
    PRED_PATH = path.join("../input/Labels_35")

    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    PRED_PATHS = np.sort([path.join(PRED_PATH, img) for img in os.listdir(PRED_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])

    test_ds = read_data(IMG_PATHS, PRED_PATHS, LABEL_PATHS)
    test_ds = mri_preprocess(test_ds)
    sagit_test = get_2D_dataset(test_ds, 0)

