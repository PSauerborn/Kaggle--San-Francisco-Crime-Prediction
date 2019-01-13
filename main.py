import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from processing import process_data



data = pd.read_csv('./data/train_formatted.csv')
target = pd.read_csv('./data/train_labels_formatted.csv')

X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.1, random_state=1, stratify=target.values)

X_valid, y_valid = X_train[750000:, :], y_train[750000:]
X_train, y_train = X_train[:750000, :], y_train[:750000]

train_set = (X_train, y_train)
validation_set = (X_valid, y_valid)

from model import Classifier

eta, n_epochs = 0.00005, 20
n_features, n_classes = X_train.shape[1], y_train.shape[1]

build = {'n_layers': 1, 'n_units': [128], 'activation_fn': [tf.nn.relu]}

model = Classifier(n_features=n_features, n_classes=n_classes, n_epochs=n_epochs, eta=eta, build=build)

model.fit(train_set=train_set, valid_set=validation_set, batch_size=250)

model.plot_train()
