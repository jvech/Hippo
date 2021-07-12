import tensorflow as tf
import nibabel as nib
import numpy as np
import os

from os import path


def read_data(img_paths, label_paths):
    def mri_read():
        for X, Y in zip(img_paths, label_paths):
            yield (tf.constant(nib.load(X).get_fdata(), "float64"),
                   tf.constant(nib.load(Y).get_fdata(), "float64"))
    ds = tf.data.Dataset.from_generator(
                mri_read,
                output_signature = (tf.TensorSpec((197, 233, 189), tf.float64), 
                                    tf.TensorSpec((197, 233, 189), tf.float64))
                )

    return ds

def mri_preprocess(ds):
    def preprocessing3D(x, y):
        ## X = (X - X_mean)/X_std
        x = (x - tf.reduce_mean(x))/ tf.math.reduce_std(x)
        y = tf.where(y!=0, 1, 0)
        return x, y

    def extract_roi(x, y, ijk=(40, 50, 30), whl=(100, 100, 100)):
        i, j, k = ijk
        w, h, l = whl
        return x[i:i+w, j:j+h, k:k+l], y[i:i+w, j:j+h, k:k+l]

    pre_ds = ds.map(lambda x, y: preprocessing3D(x, y))
    return pre_ds.map(lambda x, y: extract_roi(x, y))  

def get_2D_dataset(ds, axis):
    def get_axis_view(X, Y):
        x = []
        y = []
        for i in range(X.shape[axis]):
            x.append(X[(slice(None),)*axis + (i,)])
            y.append(Y[(slice(None),)*axis + (i,)])
        x = tf.data.Dataset.from_tensor_slices(x)
        y = tf.data.Dataset.from_tensor_slices(y)
        return tf.data.Dataset.zip((x, y))

    def mri_2D_preprocess(x, y):
        return tf.expand_dims(x, -1), tf.expand_dims(y, -1)

    ds_2D = ds.flat_map(lambda x, y: get_axis_view(x, y))
    ds_2D = ds_2D.map(lambda x, y: mri_2D_preprocess(x, y))

    return ds_2D

if __name__ == "__main__":
    IMG_PATH = path.join("../input/images_35")
    LABEL_PATH = path.join("../input/Labels_35")
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])

    test_ds = read_data(IMG_PATHS, LABEL_PATHS)
    test_ds = mri_preprocess(test_ds)
    sagit_test = get_2D_dataset(test_ds, 0)
    for X, Y in sagit_test.take(1):
        print(X.shape, Y.shape)

