import tensorflow as tf
import numpy as np


def make_gauss_kernel(kernel_size, sigma):
    _y, _x = np.mgrid[-kernel_size[1]//2 + 1: kernel_size[1]//2 + 1,
                      -kernel_size[0]//2 + 1: kernel_size[0]//2 + 1]

    _x = _x.reshape(list(_x.shape) + [1, 1])
    _y = _y.reshape(list(_y.shape) + [1, 1])
    x = tf.constant(_x, dtype=tf.float32)
    y = tf.constant(_y, dtype=tf.float32)

    g = tf.exp(-(x**2 + y**2) / (2.0*sigma ** 2), name='gauss_kernel')
    return g / tf.reduce_sum(g)


def ssim_loss(y_true, y_pred,
              L=1.0, K1=0.01, K2=0.03,
              kernel_size=(3, 3), sigma=1.0):
    with tf.name_scope('SSIM_loss'):
        bs, h, w, c = y_true.shape

        _y_true = tf.constant(np.array(y_true))
        _y_true = tf.transpose(_y_true, (0, 3, 1, 2))
        _y_true = tf.reshape(_y_true, (bs*c, h, w))
        _y_true = tf.expand_dims(_y_true, axis=-1)

        _y_pred = tf.constant(np.array(y_pred))
        _y_pred = tf.transpose(_y_pred, (0, 3, 1, 2))
        _y_pred = tf.reshape(_y_pred, (bs*c, h, w))
        _y_pred = tf.expand_dims(_y_pred, axis=-1)

        g_kernel = make_gauss_kernel(kernel_size, sigma)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        mu_true = tf.nn.conv2d(_y_true, g_kernel, strides=[1, 1, 1, 1], padding='VALID', name='mu_true')
        mu_pred = tf.nn.conv2d(_y_pred, g_kernel, strides=[1, 1, 1, 1], padding='VALID', name='mu_pred')

        mu_true_true = tf.multiply(mu_true, mu_true, name='mu_true_true')
        mu_pred_pred = tf.multiply(mu_pred, mu_pred, name='mu_pred_pred')
        mu_true_pred = tf.multiply(mu_true, mu_pred, name='mu_true_pred')

        sigma_true_true = tf.subtract(tf.nn.conv2d(_y_true * _y_true, g_kernel, strides=[1, 1, 1, 1], padding='VALID'),
                                      mu_true_true,
                                      name='sigma_true_true')
        sigma_pred_pred = tf.subtract(tf.nn.conv2d(_y_pred * _y_pred, g_kernel, strides=[1, 1, 1, 1], padding='VALID'),
                                      mu_pred_pred,
                                      name='sigma_pred_pred')
        sigma_true_pred = tf.subtract(tf.nn.conv2d(_y_true * _y_pred, g_kernel, strides=[1, 1, 1, 1], padding='VALID'),
                                      mu_true_pred,
                                      name='sigma_true_pred')

        loss = (2*mu_true_pred + C1) * (2*sigma_true_pred + C2)
        loss /= (mu_true_true + mu_pred_pred + C1) * (sigma_true_true + sigma_pred_pred + C2)
    return tf.reduce_mean(loss)


def make_gaussian_pyramid(x, max_level=5, kernel_size=(3, 3), sigma=1.0, gaussian_iteration=1):
    bs, h, w, c = x.shape
    _x = tf.transpose(x, (0, 3, 1, 2))
    _x = tf.reshape(_x, (bs*c, h, w))
    _x = tf.expand_dims(_x, axis=-1)
    pyramid = [_x]
    g_kernel = make_gauss_kernel(kernel_size, sigma)
    for level in range(max_level):
        current = pyramid[-1]
        downsampled = tf.nn.avg_pool(current, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        filtered = downsampled
        for _ in range(gaussian_iteration):
            filtered = tf.nn.conv2d(filtered, g_kernel, strides=[1, 1, 1, 1], padding='SAME')
        pyramid.append(filtered)
    return pyramid


def make_laplacian_pyramid(x, max_level=5, kernel_size=(3, 3), sigma=1.0, gaussian_iteration=1):
    g_pyr = make_gaussian_pyramid(x, max_level, kernel_size, sigma, gaussian_iteration)
    l_pyr = []
    for level in range(max_level):
        high_reso = g_pyr[level]
        low_reso = g_pyr[level+1]

        bs, h, w, c = high_reso.get_shape().as_list()
        up_low_reso = tf.image.resize_bilinear(low_reso, size=(w, h))

        diff = high_reso - up_low_reso
        l_pyr.append(diff)
    return l_pyr


if __name__ == '__main__':
    import cv2
    im = cv2.imread('./images/Lenna.png')
    im = np.expand_dims(im, 0)
    im = im.astype('float32') / 255
    sess = tf.Session()
    # l = ssim_loss(im, im)
    l = make_laplacian_pyramid(im)
    tf.summary.FileWriter('logs', graph=sess.graph)
