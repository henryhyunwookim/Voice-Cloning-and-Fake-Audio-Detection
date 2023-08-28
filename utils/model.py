import os
import json
from datetime import datetime
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.statistics import f1_score


def compile_sequential_model(num_layers, num_labels, units_list, activation_list,
                            input_shape, dropout_rate, optimizer, loss, metrics):
    # units: Positive integer, dimensionality of the output space.
    model=Sequential()

    for i, units, activation in zip(range(num_layers), units_list, activation_list):
        if i == 0: # The first layer
            model.add(Dense(units, activation=activation, input_shape=input_shape))
            model.add(Dropout(dropout_rate))

        elif i == num_layers - 1: # The last layer
            model.add(Dense(num_labels, activation=activation))

        else:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(dropout_rate))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def compile_and_fit_sequential_model(best_model_path, best_result_path,
                                    num_layers, num_labels, units_list, activation_list,
                                    input_shape, dropout_rate, optimizer, loss, metrics,
                                    train_X, train_y, test_X, test_y,
                                    num_batch_size, num_epochs, checkpointer=None, save_model=True, overwrite=False
):
    if os.path.exists(best_model_path) and os.path.exists(best_result_path) and not overwrite:
        print('Load saved model and results from drive.')
        model = load_model(best_model_path)
        history = json.load(open(best_result_path, 'r'))

    else:
        if overwrite:
            print('Overwriting existing files in drive.')
        else:
            print('No model exsits in drive. Creating one.')
        model = compile_sequential_model(
                        num_layers, num_labels, units_list, activation_list,
                        input_shape, dropout_rate, optimizer, loss, metrics)
        print(model.summary())

        # checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=2, save_best_only=True)

        start = datetime.now()

        if checkpointer != None:
            history = model.fit(
                    train_X,\
                    train_y,
                    batch_size=num_batch_size,
                    epochs=num_epochs,
                    validation_data=(test_X, test_y),
                    callbacks=[checkpointer],
                    verbose=2
                    )
        else:
            history = model.fit(
                    train_X,\
                    train_y,
                    batch_size=num_batch_size,
                    epochs=num_epochs,
                    validation_data=(test_X, test_y),
                    # callbacks=[checkpointer],
                    verbose=2
                    )
        # By default verbose = 1,
        # verbose = 1, which includes both progress bar and one line per epoch
        # verbose = 0, means silent
        # verbose = 2, one line per epoch i.e. epoch no./total no. of epochs

        duration = datetime.now() - start
        print(f'\nTraining completed in {duration}.')

        # Convert the tf.keras.callbacks.History object into a dictionary.
        history = {**history.params, 'epoch': history.epoch, **history.history}

        for k, v in history.items():
            if 'f1_score' in k:
                history[k] = [float(np.mean(f1_score)) for f1_score in v]

        if save_model and overwrite:
            # Save the model and results
            model.save(best_model_path)
            json.dump(history, open(best_result_path, 'w'))
            print('Model and results saved in drive.')

    return model, history