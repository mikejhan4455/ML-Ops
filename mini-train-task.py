# experiense of mlflow MLops
# Reference: https://tinyurl.com/2yptg2m7


# Title: Simple MNIST convnet
# Description: A simple convnet that achieves ~99% test accuracy on MNIST.
# Setup
from distutils.command.build import build
import numpy as np
from tensorflow import keras
from keras import layers
import mlflow
from kerastuner.tuners import RandomSearch, BayesianOptimization

# Prepare the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# find mlflow run_id
def find_run_id(name):
    histories = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{name}'")
    if len(histories["run_id"]) > 1:
        raise NotImplementedError("to many run_id with the same runName.")
    elif len(histories["run_id"]) == 1:
        run_id = histories["run_id"][0]
    else:
        run_id = None

    return run_id


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units_layer1", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(
        layers.Dense(
            units=hp.Int("units_layer2", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(
        layers.Dropout(
            rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.8, step=0.1)
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        #         loss='categorical_crossentropy',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
## Build the model
model = build_model()
print(model.summary())

# mlflow config
mlflow.set_tracking_uri("http://163.13.128.208:5000")
mlflow.set_experiment("mini-train-task-mnist")

## Train the model
batch_size = 128 * 2
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# run with mlflow
RUN_NAME = "run_02"

with mlflow.start_run(find_run_id(RUN_NAME), run_name=RUN_NAME):
    mlflow.keras.autolog()
    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )
