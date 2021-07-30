import tensorflow as tf
import numpy as np

def jaccard(y_true,y_pred):
    inter = np.sum((y_true*y_pred)!= 0)
    union = np.sum(y_true!=0) + np.sum(y_pred!=0) - inter
    if union == 0: union=1e-6
    return inter/union 

def dice(y_true,y_pred):
    inter = np.sum((y_true*y_pred)!= 0)
    suma = np.sum(y_true!=0) + np.sum(y_pred!=0)
    if suma == 0: suma=1e-6
    return 2*inter/suma

def precision(y_true,y_pred):
    inter = np.sum((y_true*y_pred)!= 0)
    estimate = np.sum(y_pred!=0)
    if estimate == 0: estimate=1e-6
    return inter/estimate

def recall(y_true,y_pred):
    inter = np.sum((y_true*y_pred)!= 0)
    mascara = np.sum(y_true!=0)
    if mascara == 0: mascara=1e-6
    return inter/mascara

def get_scores(y_true, y_pred):
    return [mt(y_true, y_pred) for mt in [jaccard, dice, precision, recall]]

def scores2D(test_ds, model):
    scores = []
    for x, y in test_ds:
        x = tf.expand_dims(x, 0)
        y_pred = model.predict(x)
        y_pred = tf.where(y_pred > 0.5, 1, 0)

        scores.append(get_scores(y[...,0], y_pred[0,...,0]))
    return np.array(scores).mean(axis=0), np.array(scores).std(axis=0)

def scores3D(test_ds, model):
    scores = []
    for X, Y in test_ds:
        x = tf.expand_dims(X, -1)
        y_pred = model.predict(x)
        y_pred = tf.where(y_pred > 0.5, 1, 0)

        scores.append(get_scores(Y, y_pred[...,0]))
    return np.array(scores).mean(axis=0), np.array(scores).std(axis=0)

def print_scores(name, scores):
    print(name.upper())
    for i,mt in enumerate(['Jaccar','Dice','Precision','Recall']):
        print(f'{mt:<10}:{100*scores[0][i]:2.2f} +/- {100*scores[1][i]:2.2f}')
    print('')
