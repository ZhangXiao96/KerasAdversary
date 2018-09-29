import numpy as np


def get_shuffle_indices(data_size):
    return np.random.permutation(np.arange(data_size))


def batch_iter(data, batchsize, shuffle=True):
    data = np.array(list(data))
    data_size = data.shape[0]
    num_batches = int((data_size-1)/batchsize) + 1
    # Shuffle the data
    if shuffle:
        shuffle_indices = get_shuffle_indices(data_size)
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches):
        start_index = batch_num * batchsize
        end_index = min((batch_num + 1) * batchsize, data_size)
        yield shuffled_data[start_index:end_index]


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError('The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = get_shuffle_indices(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split)-1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list


def linear_interpolation(x_start, x_end, num):
    step = (x_end-x_start)/(num+1.)
    x = [x_start + i * step for i in range(num+2)]
    step = np.reshape(step, newshape=[-1, ])
    return x, np.linalg.norm(step, ord=2)


def images_fft(image_list):
    images = np.array(image_list)
    f_images = np.fft.fft2(images, axes=(1, 2))
    f_images = np.fft.fftshift(f_images, axes=(1, 2))
    A = np.abs(f_images)
    phi = np.angle(f_images)
    return A, phi