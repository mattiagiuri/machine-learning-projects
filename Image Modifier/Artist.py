import os
import numpy as np
import scipy.io
import tensorflow as tf
from sys import stderr
import time
import matplotlib.image as im

ITERATIONS = 2000

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0),
]

NOISE_RATIO = 0.6
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
COLOR_CHANNELS = 3


def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):

    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def style_loss_func(sess, model):

    def _gram_matrix(F, N, M):

        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss


def content_loss_func(sess, model):
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])


def imgunprocess(image):
    temp = image + VGG19_mean
    return temp[0]


def _conv2d_relu(prev_layer, n_layer):
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, bias.size))
    conv2d = tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b
    return tf.nn.relu(conv2d)


def _avgpool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def to_rgb(image):
    w, h = image.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    return ret


def imgpreprocess(image):
    image = image[np.newaxis,:,:,:]
    return image - VGG19_mean


file_content_image = 'houses-of-parliament-at-night.jpg'
file_style_image = 'starry-night.jpg'

ALPHA = 100
BETA = 5

path_VGG19 = 'imagenet-vgg-verydeep-19.mat'
VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

n_iterations_checkpoint = 100
path_output = 'output directory'

if not os.path.exists(path_output):
    os.mkdir(path_output)

img_content = im.imread(file_content_image)
img_style = im.imread(file_style_image)


if len(img_content.shape) == 2:
    img_content = to_rgb(img_content)

if len(img_style.shape) == 2:
    img_style = to_rgb(img_style)


img_content = imgpreprocess(img_content)
img_style = imgpreprocess(img_style)


img_initial = generate_noise_image(img_content)
VGG19 = scipy.io.loadmat(path_VGG19)
VGG19_layers = VGG19['layers'][0]

with tf.compat.v1.Session() as sess:
    a, h, w, d = img_content.shape
    net = {'input': tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))}
    net['conv1_1'] = _conv2d_relu(net['input'], 0)
    net['conv1_2'] = _conv2d_relu(net['conv1_1'], 2)
    net['avgpool1'] = _avgpool(net['conv1_2'])
    net['conv2_1'] = _conv2d_relu(net['avgpool1'], 5)
    net['conv2_2'] = _conv2d_relu(net['conv2_1'], 7)
    net['avgpool2'] = _avgpool(net['conv2_2'])
    net['conv3_1'] = _conv2d_relu(net['avgpool2'], 10)
    net['conv3_2'] = _conv2d_relu(net['conv3_1'], 12)
    net['conv3_3'] = _conv2d_relu(net['conv3_2'], 14)
    net['conv3_4'] = _conv2d_relu(net['conv3_3'], 16)
    net['avgpool3'] = _avgpool(net['conv3_4'])
    net['conv4_1'] = _conv2d_relu(net['avgpool3'], 19)
    net['conv4_2'] = _conv2d_relu(net['conv4_1'], 21)
    net['conv4_3'] = _conv2d_relu(net['conv4_2'], 23)
    net['conv4_4'] = _conv2d_relu(net['conv4_3'], 25)
    net['avgpool4'] = _avgpool(net['conv4_4'])
    net['conv5_1'] = _conv2d_relu(net['avgpool4'], 28)
    net['conv5_2'] = _conv2d_relu(net['conv5_1'], 30)
    net['conv5_3'] = _conv2d_relu(net['conv5_2'], 32)
    net['conv5_4'] = _conv2d_relu(net['conv5_3'], 34)
    net['avgpool5'] = _avgpool(net['conv5_4'])

    sess.run(net['input'].assign(img_content))
    content_loss = content_loss_func(sess, net)

    sess.run(net['input'].assign(img_style))
    style_loss = style_loss_func(sess, net)

    total_loss = BETA * content_loss + ALPHA * style_loss

    optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)
    init_op = tf.compat.v1.initialize_all_variables()
    sess.run(init_op)
    sess.run(net['input'].assign(img_initial))

    for i in range(1, ITERATIONS + 1):
        sess.run(train_step)

        if i % 100 == 0:
            stderr.write('Iteration %d/%d\n' % (i, ITERATIONS))
            img_output = sess.run(net['input'])
            img_output = imgunprocess(img_output)
            print('sum : ', sess.run(tf.reduce_sum(img_output)))
            print('cost: ', sess.run(total_loss))
            img_output = np.clip(img_output, 0, 255).astype('uint8')
            timestr = time.strftime("%Y%m%d_%H%M%S")
            output_file = path_output + '/' + timestr + '_' + '%s.jpg' % (i * n_iterations_checkpoint)
            im.imsave(output_file, img_output)
            print('saved')
