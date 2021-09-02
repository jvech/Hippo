# Hippocampus Segmentation
## INSTALLATION
```
git clone https://github.com/jvech/Hippo.git
cd src
```

## USAGE
### Training

```
Usage: 
    train.py [options] <imgs> <masks> (sagital | coronal | axial) 
    train.py [options] <imgs> <masks> <preds> (sagital | coronal | axial)
    train.py [options] retrain <model> <imgs> <masks> (sagital | coronal | axial)
    train.py [options] retrain <model> <imgs> <masks> <preds> (sagital | coronal | axial)
    train.py (-h | --help)

Options:
    -h --help           Show this message
    -H --history        Plot the history of the model's performance
    -o <file>           Save the trained model into <file> [default: ./model.h5].
    --batch <int>       Batch size [default: 8]
    --epochs <int>      Number of epochs [default: 40]
```

### Model Inference

```
Usage: 
    predict.py [options] <img> <models_folder>
    predict.py (-h | --help)

Options:
    -h --help           Show this message
    -s --seg-only       Use only the Segmentation models
    -o <file>           Place de output into <file> [default: ./prediction.nii]
    -v --verbose        Show the results of the prediction
```

### Evaluation

```
Usage: 
    eval.py [options] <imgs> <masks> <models_path>
    eval.py (-h | --help)

Options:
    -h --help           Show this message
    -s --seg-only       Use only the Segmentation models
```
