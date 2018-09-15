"""
Last updated: 2018.09.14
This file is used to provide some methods for visualization.
Please email to xiao_zhang@hust.edu.cn if you have any questions.
"""
import matplotlib.pyplot as plt
from scipy.misc import imshow
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_loss(train_loss, val_loss, title=None):
    if not title:
        title = 'Train Loss and Val Loss'
    if not len(train_loss) == len(val_loss):
        raise TypeError('The lengths of train_loss and val_loss should be the same!')

    x = range(len(train_loss))
    plt.plot(x, train_loss, label='train')
    plt.plot(x, val_loss, label='val')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)


def show_x_and_adversarial_x(x_list, adv_x_list, show_dif=True):
    """
    This function is used to plot x, adversarial x and the normalized (adv_x-x).
    :param x_list: the list of the normal images.
    :param adv_x_list: the list of the adversarial images.
    :param show_dif: Plot the normalized (adv_x-x) if True.
    """
    if not len(x_list) == len(adv_x_list):
        raise TypeError('The lengths of x_list and adv_x_list are not the same.')
    x = np.concatenate(x_list, axis=1)
    adv_x = np.concatenate(adv_x_list, axis=1)
    if show_dif:
        dif = adv_x - x
        dif_heat = (dif-np.min(dif))/(np.max(dif)-np.min(dif))
        merge_image = np.concatenate([x, adv_x, dif_heat], axis=0)
    else:
        merge_image = np.concatenate([x, adv_x], axis=0)
    imshow(merge_image)


def plot_confusion_matrix(true_y, pred_y):
    # labels = list(np.unique(true_y))
    # cm = confusion_matrix(y_true=true_y, y_pred=pred_y, labels=labels)
    #TODO: plot confusion_matrix
    pass