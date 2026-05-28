import numpy as np


def cutcat(data_1, label_1, data_2, label_2, num_classes, ratio=8):
    c, t = data_1.shape[1], data_1.shape[2]
    max_length = max(1, t // ratio)
    length = np.random.randint(1, max_length + 1)

    center = np.random.randint(t)
    x1 = np.clip(center - length // 2, 0, t)
    x2 = np.clip(center + length // 2, 0, t)

    mask_1 = np.ones((1, c, t), np.float32)
    mask_1[:, :, x1:x2] = 0.0
    mixed_1 = data_1 * mask_1

    mask_2 = np.zeros((1, c, t), np.float32)
    mask_2[:, :, x1:x2] = 1.0
    mixed_2 = data_2 * mask_2

    data = mixed_1 + mixed_2

    one_hot_label_1 = np.zeros(num_classes, np.float32)
    one_hot_label_1[label_1] = 1.0
    one_hot_label_2 = np.zeros(num_classes, np.float32)
    one_hot_label_2[label_2] = 1.0

    lamb = (x2 - x1) / t
    label = (1.0 - lamb) * one_hot_label_1 + lamb * one_hot_label_2
    return data, label


def cutcat_2(data_1, label_1, data_2, label_2, num_classes, ratio=8):
    one_hot_label_1 = np.zeros(num_classes, np.float32)
    one_hot_label_1[label_1] = 1.0
    return data_1, one_hot_label_1
