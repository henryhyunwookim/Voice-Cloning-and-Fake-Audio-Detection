from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


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