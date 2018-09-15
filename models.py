from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Activation, Input


def DensNet(input_shape, nb_class, hidden_units):
    input_x = Input(shape=input_shape)
    x = Flatten()(input_x)
    for unit in hidden_units:
        x = Dense(units=unit, activation='relu')(x)
    logits = Dense(units=nb_class)(x)
    y = Activation(activation='softmax')(logits)
    return Model(inputs=input_x, outputs=y)