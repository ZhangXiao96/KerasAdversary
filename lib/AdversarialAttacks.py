"""
Last updated: 2018.09.14
This file is used to perform some adversarial attack methods on the deep learning models built with Keras.
Now involve:
    white-box attack:
        FGSM, BIM, C&W.
    black_box attack:

We will update the methods irregularly.
Please email to xiao_zhang@hust.edu.cn if you have any questions.
"""
from keras import backend as K
from lib import utils
from tqdm import tqdm
import numpy as np


class WhiteBoxAttacks(object):
    """
    This class provides a simple interface to perform target white-box attacks on keras models.
    For example, if you want to perform FGSM, you can simply use

        AttackAgent = WhiteBoxAttacks(target_model, session)
        adv_x = AttackAgent.fgsm(x, y, epsilon=0.1)

    to generate adversarial examples of x.
    """
    def __init__(self, model, sess):
        """
        To generate the White-box Attack Agent.
        RNN is not supported now.
        :param model: the target model which should have the input tensor, the target tensor and the loss tensor.
        :param sess: the tensorflow session.
        """
        self.model = model
        self.input_tensor = model.inputs[0]
        self.target_tensor = model.targets[0]
        self.loss_tensor = model.total_loss
        self.gradient_tensor = K.gradients(self.loss_tensor, self.input_tensor)[0]
        self.sess = sess
        self._sample_weights = model.sample_weights[0]

    def get_model(self):
        return self.model

    def get_sess(self):
        return self.sess

    def _get_batch_gradients(self, x_batch, y_batch):
        feed_dict = {
            self.input_tensor: x_batch,
            self.target_tensor: y_batch,
            self._sample_weights: np.ones((len(y_batch),))
        }
        gradient_batch = self.sess.run(self.gradient_tensor, feed_dict=feed_dict)
        return gradient_batch

    def get_gradients(self, x, y, batch_size=256):
        """
        This function is used to get the gradients \Delta_{x}Loss(x,y;\theta)
        :param x: the normal examples
        :param y: the labels of x
        :param batch_size: batch size
        :return: gradients
        """
        K.set_learning_phase(0)
        gradients = []
        data = zip(x, y)
        batches = list(utils.batch_iter(data, batchsize=batch_size, shuffle=False))
        for batch in tqdm(batches):
            x_batch, y_batch = zip(*batch)
            gradient_batch = self._get_batch_gradients(x_batch=x_batch, y_batch=y_batch)
            gradients.append(gradient_batch)
        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def ag(self, x, y, epsilon=0.1, ord=2, batch_size=256, clip_min=None, clip_max=None):
        """
        Add Gradients (ag).
        Just add the gradients whose ord norm is epsilon (fixed).
        :param x: the normal examples
        :param y: the labels of x
        :param epsilon: the limit of the norm of the gradient.
        :param ord: the ord of the norm
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        K.set_learning_phase(0)
        gradients = self.get_gradients(x, y, batch_size=batch_size)
        adv_flat = np.reshape(gradients, newshape=[gradients.shape[0], -1])
        norms = np.linalg.norm(adv_flat, ord=ord, axis=1, keepdims=True)

        adv_noise = np.reshape(epsilon * adv_flat / norms, newshape=gradients.shape)
        adv_x = x + adv_noise
        if clip_min is not None or clip_max is not None:
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        return adv_x

    def fgsm(self, x, y, epsilon=0.1, batch_size=256, clip_min=None, clip_max=None):
        """
        Fast Gradient Sign Method (FGSM).
        The original paper can be found at: https://arxiv.org/abs/1412.6572
        @Article{FGSM,
          author        = {Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
          title         = {Explaining and Harnessing Adversarial Examples},
          journal       = {CoRR},
          year          = {2014},
          volume        = {abs/1412.6572},
          archiveprefix = {arXiv},
          eprint        = {1412.6572},
          url           = {http://arxiv.org/abs/1412.6572},
        }
        :param x: the normal examples
        :param y: the labels of x
        :param epsilon: the limit of the permutation
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        K.set_learning_phase(0)
        gradients = self.get_gradients(x, y, batch_size=batch_size)
        adv_noise = epsilon * np.sign(gradients)
        adv_x = x + adv_noise
        if clip_min is not None or clip_max is not None:
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        return adv_x

    def bim(self, x, y, epsilon=0.1, iterations=3, batch_size=256, clip_min=None, clip_max=None):
        """
        Basic Iterative Method (BIM).
        The original paper can be found at: https://arxiv.org/abs/1607.02533
        @Article{BIM,
          author        = {Alexey Kurakin and Ian J. Goodfellow and Samy Bengio},
          title         = {Adversarial examples in the physical world},
          journal       = {CoRR},
          year          = {2016},
          volume        = {abs/1607.02533},
          archiveprefix = {arXiv},
          eprint        = {1607.02533},
          url           = {http://arxiv.org/abs/1607.02533},
        }
        :param x: the normal examples
        :param y: the labels of x
        :param epsilon: the limit of the permutation
        :param iterations: number of attack iterations.
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        K.set_learning_phase(0)
        adv_x = x
        for iteration in range(iterations):
            print('Performing BIM: {}/{} iterations'.format(iteration+1, iterations))
            adv_x = self.fgsm(adv_x, y, epsilon=epsilon, batch_size=batch_size, clip_min=clip_min, clip_max=clip_max)
        return adv_x

    def CarliniAndWagner(self, x, y):
        """
        Carlini & Wagner (C&W).
        The original paper can be found at: https://arxiv.org/abs/1608.04644
        @Article{CandW,
          author  = {Nicholas Carlini and David A. Wagner},
          title   = {Towards Evaluating the Robustness of Neural Networks},
          journal = {CoRR},
          year    = {2016},
          volume  = {abs/1608.04644},
          url     = {https://arxiv.org/abs/1608.04644},
        }
        """
        K.set_learning_phase(0)
        pass