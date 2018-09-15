from lib.AdversarialAttacks import WhiteBoxAttacks
import models
from lib import visualization
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.datasets import mnist
import keras.backend as K
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

runs_dir = 'runs'
ckpt_name = 'mnist_DenseNet'
model_file = 'model.h'

ckpt_dir = os.path.join(runs_dir, ckpt_name)
model_path = os.path.join(ckpt_dir, model_file)
if not os.path.exists(model_path):
    raise TypeError('No such file-{}!'.format(model_path))

(train_x, train_y), (test_x, test_y) = mnist.load_data()
test_x = test_x/255.
test_y = test_y.reshape([-1, 1])

x_shape = test_x.shape[1:]
nb_class = len(np.unique(test_y))
hidden_units = [100, 100]
target_model = models.DensNet(input_shape=x_shape, nb_class=nb_class, hidden_units=hidden_units)
target_model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['acc'])
target_model.load_weights(model_path)

attack_agent = WhiteBoxAttacks(target_model, sess=K.get_session())
adv_x = attack_agent.bim(test_x, test_y, epsilon=0.03, iterations=3, clip_min=0, clip_max=1)
# adv_x = attack_agent.fgsm(test_x, test_y, epsilon=0.1, clip_min=0, clip_max=1)

loss, acc = target_model.evaluate(adv_x, test_y, verbose=0)
print('adversarial test: loss={}, acc={}'.format(loss, acc))
loss, acc = target_model.evaluate(test_x, test_y, verbose=0)
print('clean test: loss={}, acc={}'.format(loss, acc))

visualization.show_x_and_adversarial_x(x_list=test_x[:10], adv_x_list=adv_x[:10], show_dif=True)
