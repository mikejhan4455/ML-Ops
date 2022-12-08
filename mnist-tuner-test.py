"""mlflow integration for KerasTuner
Copyright: Soundsensing AS, 2021
License: MIT
"""
import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import uuid

import structlog

log = structlog.get_logger()

import mlflow
import keras_tuner


def get_run_id(run):
    if run is None:
        return None

    return run.info.run_id


class MlflowLogger(object):
    """KerasTuner Logger for integrating with mlflow
    Each KerasTuner trial is a parent mlflow run,
    and then each execution is a child
    XXX: assumes that executions are done sequentially and non-concurrently
    """

    def __init__(self):
        self.search_run = None
        self.search_id = None
        self.trial_run = None
        self.trial_id = None
        self.trial_state = None
        self.execution_run = None
        self.execution_id = 0

    def register_tuner(self, tuner_state):
        """Called at start of search"""

        log.debug("mlflow-logger-search-start")

        self.search_id = str(uuid.uuid4())
        # Register a top-level run
        self.search_run = mlflow.start_run(
            nested=False, run_name=f"search-{self.search_id[0:8]}"
        )

    def exit(self):
        """Called at end of a search"""

        log.debug("mlflow-logger-search-end")
        self.seach_run = None
        self.search_id = None

    def register_trial(self, trial_id, trial_state):
        """Called at beginning of trial"""

        log.debug(
            "mlflow-logger-trial-start",
            trial_id=trial_id,
            active_run_id=get_run_id(mlflow.active_run()),
        )

        assert self.search_run is not None
        assert self.trial_run is None
        assert self.execution_run is None
        assert self.execution_id == 0

        self.trial_id = trial_id
        self.trial_state = trial_state

        # Start a new run, under the search run
        self.trial_run = mlflow.start_run(
            nested=True, run_name=f"trial-{self.trial_id[0:8]}-{self.search_id[0:8]}"
        )

        # For now, only register these on each execution
        # hyperparams = self.trial_state['hyperparameters']['values']
        # mlflow.log_params(hyperparams)

    def report_trial_state(self, trial_id, trial_state):
        """Called at end of trial"""

        log.debug(
            "mlflow-logger-trial-end",
            trial_id=trial_id,
            active_run_id=get_run_id(mlflow.active_run()),
        )

        assert self.search_run is not None
        assert self.trial_run is not None
        assert self.execution_run is None

        # Start a new run, under the search run
        mlflow.end_run()  ## XXX: no way to specify the id?

        self.trial_run = None
        self.trial_id = None
        self.trial_state = None
        self.execution_id = 0

    def register_execution(self):
        log.debug(
            "mlflow-logger-execution-start",
            active_run_id=get_run_id(mlflow.active_run()),
        )

        assert self.search_run is not None
        assert self.trial_run is not None
        assert self.execution_run is None

        self.execution_run = mlflow.start_run(
            nested=True,
            run_name=f"exec-{self.execution_id}-{self.trial_id[0:8]}-{self.search_id[0:8]}",
        )
        self.execution_id += 1

        # register hyperparameters from the trial
        hyperparams = self.trial_state["hyperparameters"]["values"]
        mlflow.log_params(hyperparams)

    def report_execution_state(self, histories):
        log.debug(
            "mlflow-logger-execution-end",
            active_run_id=get_run_id(mlflow.active_run()),
        )
        assert self.search_run is not None
        assert self.trial_run is not None
        assert self.execution_run is not None

        mlflow.end_run()  ## XXX: no way to specify the id?

        self.execution_run = None


class FakeHistories:
    def __init__(self, metrics={}):
        self.history = metrics


class LoggerTunerMixin:
    def __init__(self, *args, **kwargs):
        if kwargs.get("logger") is None:
            kwargs["logger"] = MlflowLogger()

        self.on_exception = kwargs.get("on_exception", "pass")

        return super(LoggerTunerMixin, self).__init__(*args, **kwargs)

    # Hack in registration for each model training "execution"
    def _build_and_fit_model(self, trial, *args, **kwargs):

        # log start of execution
        if self.logger:
            self.logger.register_execution()

        histories = None
        try:
            # call the original function
            histories = super(LoggerTunerMixin, self)._build_and_fit_model(
                trial, *args, **kwargs
            )
        except Exception as e:
            if self.on_exception == "pass":
                o = self.oracle.objective
                value = float("inf") if o.direction == "min" else float("-inf")
                histories = FakeHistories({o.name: value})
            else:
                raise e

        # log end of execution
        if self.logger:
            self.logger.report_execution_state(histories)

        return histories


# Integrate with keras tuners
class RandomSearch(LoggerTunerMixin, keras_tuner.RandomSearch):
    pass


class BayesianOptimization(LoggerTunerMixin, keras_tuner.BayesianOptimization):
    pass


class SklearnTuner(LoggerTunerMixin, keras_tuner.SklearnTuner):
    pass


class Hyperband(LoggerTunerMixin, keras_tuner.Hyperband):
    pass


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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Title: Simple MNIST convnet

# Prepare the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 10
max_trials = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255, -1)


## Train the model
# mlflow config
mlflow.set_tracking_uri("http://163.13.128.208:5000")
mlflow.set_experiment("mini-train-task-mnist")


# mlflow.keras.autolog()

# 選用隨機搜索
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",  # 優化目標爲精度'val_accuracy'（最小化目標）
    max_trials=max_trials,
    executions_per_trial=3,  # 每次試驗訓練模型三次
    directory="my_dir",
    logger=MlflowLogger(),
)

tuner.search(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_test, y_test),
)


# ## Evaluate the trained model
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
