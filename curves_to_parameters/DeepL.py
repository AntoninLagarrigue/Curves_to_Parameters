import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import json
import jsonpickle
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from IPython.display import clear_output
import pprint

pp = pprint.PrettyPrinter()




gen_size = 200000
minA = 0.5
maxA = 1
min_omega = 0.5
max_omega = 0.8
sequenceLength = 100
step_duration = 0.025  # in second

units = 64



def get_uncompiled_model():
    model = keras.Sequential()
    model.add(layers.LSTM(units, input_shape=(sequenceLength, 1)))
    #model.add(layers.Dense(units, input_shape=(sequenceLength, 1)))
    model.add(layers.Dense(2))
    return model


def get_compiled_model(optimizer, loss):
    model = get_uncompiled_model()
    model.build()
    model.summary()
    model.compile(
        optimizer=optimizer,
        loss=loss,
    )
    return model




class Plot_Losses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();


# high-level training API Keras

def train_keras(batch_size: int, epochs: int, model, X, Y):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-3 less"
            min_delta=1e-4,
            # "no longer improving" being further defined as "for at least 3 epochs"
            patience=3,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-loss={loss:.4f}", save_freq=5000
        ),
        Plot_Losses()
    ]

    history = model.fit(
        x=X,
        y=Y,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
    )

    print(history.history['loss'])

    return history, history.history['loss']





class Model_Results:
    def __init__(self,
                 name,
                 losses=None,
                 gen_parameters=[200000, 0.5, 1, 0.5, 0.8, 100, 0.025],
                 hyperparameters=[110, ]
                 ):
        super().__init__()
        self.name = name
        self.losses = losses
        self.gen_parameters = gen_parameters
        self.hyperparameters = hyperparameters

    def save_mr(self, model):

        if not os.path.exists("./Models_Sinusoids/" + self.name):
            os.mkdir("./Models_Sinusoids/" + self.name)

        data_file = os.path.join("./Models_Sinusoids/" + self.name, self.name + '.txt')
        with open(data_file, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        encoded = jsonpickle.encode(self)
        data_file = os.path.join("./Models_Sinusoids/" + self.name, self.name + '.json')
        with open(data_file, 'w') as f:
            json.dump(encoded, f)

        model.save('./Models_Sinusoids/' + self.name + '/' + self.name)

        f.close()

    def load_mr(self):

        if not os.path.exists("./Models_Sinusoids/" + self.name):
            os.mkdir("./Models_Sinusoids/" + self.name)

        data_file = os.path.join("./Models_Sinusoids/" + self.name, self.name + '.json')
        with open(data_file, 'r') as f:
            loaded = json.load(f)
            decoded = jsonpickle.decode(loaded)

        self.name = decoded.name
        self.losses = decoded.losses
        self.gen_parameters = decoded.gen_parameters
        self.hyperparameters = decoded.hyperparameters

        data_file = os.path.join("./Models_Sinusoids/" + self.name, self.name + '.txt')
        with open(data_file, 'r') as f:
            pp.pprint(f.readlines())

        model = keras.models.load_model('./Models_Sinusoids/' + self.name + '/' + self.name)

        f.close()

        return decoded, model





# low-level tensorflow training functions


@tf.function
def train_step(x, y, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value



def train_loop(epochs: int, batch_size : int, X, Y, model, loss_fn, optimizer):
    losses = []

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for i in range(0, gen_size, batch_size):
            x_batch_train = X[i:i + batch_size, :, :]
            y_batch_train = Y[i:i + batch_size, :, :]

            loss_value = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer)
            losses.append(loss_value)
            # Log every 200 batches.
            if i % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (i, float(loss_value))
                )
                print("Seen so far: %d samples" % ((i + 1) * 64))

        print("Time taken: %.2fs" % (time.time() - start_time))

    plt.plot(losses, label="loss")
    plt.legend()
    plt.show()