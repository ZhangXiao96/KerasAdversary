from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from lib import utils
import models
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

runs_dir = 'runs'
ckpt_name = 'mnist_DenseNet'
model_file = 'model.h'

ckpt_dir = os.path.join(runs_dir, ckpt_name)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
model_path = os.path.join(ckpt_dir, model_file)

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x/255.
test_x = test_x/255.

x_shape = train_x.shape[1:]
nb_class = len(np.unique(test_y))
hidden_units = [100, 100]
model = models.DensNet(input_shape=x_shape, nb_class=nb_class, hidden_units=hidden_units)
model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', save_best_only=True)

shuffle_indices = utils.get_shuffle_indices(train_y.shape[0])
train_x = train_x[shuffle_indices]
train_y = train_y[shuffle_indices]
hist = model.fit(
    x=train_x, y=train_y,
    batch_size=256, epochs=100,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stop, model_checkpoint]
)
np.savez(os.path.join(ckpt_dir, 'hist.npz'), hist=hist.history)

model.load_weights(model_path)
loss, acc = model.evaluate(test_x, test_y, verbose=0)
print('test: loss={}, acc={}'.format(loss, acc))