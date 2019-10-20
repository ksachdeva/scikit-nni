import os
import glob
import numpy as np

from absl import logging
from sklearn.utils import shuffle

def _build_ds(npz_files):
    X = []
    y = []

    for f in npz_files:
        file_content = np.load(f, allow_pickle=True)
        dataset = file_content['dataset']
        X.extend(dataset)
        label_id = int(os.path.basename(f).replace('.npz', ''))
        for i in range(0, len(dataset)):
            y.append(label_id)

    return np.array(X), np.array(y)


class NpzClassificationSource(object):
    def __init__(self):
        pass

    def __call__(self, dir_path, shuffle_dataset=True):

        if dir_path is None:
            logging.error("dir_path must be a valid directory path")
            raise ValueError("dir_path must be a valid directory path")

        if not os.path.exists(dir_path):
            logging.error("dir_path is invalid !")
            raise ValueError("dir_path is invalid !")

        train_npz_files = glob.glob(os.path.join(dir_path, 'train', "*.npz"))
        test_npz_files = glob.glob(os.path.join(dir_path, 'test', "*.npz"))

        if train_npz_files is None:
            logging.error("Could not find any npz in the train directory")
            raise ValueError("Could not find any npz in the train directory")

        if test_npz_files is None:
            logging.error("Could not find any npz in the test directory")
            raise ValueError("Could not find any npz in the test directory")

        X_train, y_train = _build_ds(train_npz_files)
        X_test, y_test = _build_ds(test_npz_files)

        logging.debug(f"X_train shape - {X_train.shape}")
        logging.debug(f"y_train shape - {y_train.shape}")
        logging.debug(f"X_test shape - {X_test.shape}")
        logging.debug(f"y_test shape - {y_test.shape}")

        # Suffle only the training data set
        if shuffle_dataset:
            X_train, y_train = shuffle(X_train, y_train, random_state=2012)

        return (X_train, y_train), (X_test, y_test)
