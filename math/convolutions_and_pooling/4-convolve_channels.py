#!/usr/bin/env python3
"""Performs a convolution on images with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray of shape (m, h, w, c)
        kernel: numpy.ndarray of shape (kh, kw, c)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("kernel channels must match image channels")

    if padding == 'same':
        ph = int(np.ceil((((h - 1) * sh + kh - h) / 2)))
        pw = int(np.ceil((((w - 1) * sw + kw - w) / 2)))
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            current = padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]
            output[:, i, j] = np.sum(current * kernel, axis=(1, 2, 3))

    return output
