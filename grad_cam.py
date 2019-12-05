"""
Code modified from
https://github.com/AdamLabISU/3DGradCAM/blob/master/GradCAM.py
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.framework import ops
from scipy.ndimage.interpolation import zoom
from tensorflow.python.keras.activations import relu
import tempfile
import matplotlib.pyplot as plt
import os
import cv2

from utils.model_utils import _load_model

input_shape = (200, 1024, 200, 1)


def loss_calculation(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot((category_index), nb_classes))


def loss_calculation_shape(input_shape):
    return input_shape


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def prepareGradCAM(input_model, explanation_catagory, nb_classes):
    loss_function = lambda x: loss_calculation(x, explanation_catagory, nb_classes)
    loss_layer = Lambda(loss_function, output_shape=loss_calculation_shape, name='new_loss')(input_model.layers[-1].output)
    model = Model(inputs=input_model.inputs, outputs=loss_layer)
    #  use the loss from the layer before softmax. As best practices
    loss = K.sum(model.layers[-1].output)
    # last fully Convolutional layer to use for computing GradCAM
    conv_output = model.layers[-6].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])

    return gradient_function


def registerGradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def compileSaliencyFunction(model, activation, activation_layer=-5):
    guidedModel = modifyBackprop(model, 'GuidedBackProp', activation)
    input_img = guidedModel.input
    layer_output = guidedModel.layers[activation_layer].output
    saliency = K.gradients(K.sum(layer_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modifyBackprop(model, name, activation):
    registerGradient()
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == relu:
                layer.activation = tf.nn.relu
    #   Popping the softmax layer as it creates ambiguity in the explanation
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def GradCAM(gradient_function, input_file):
    output, grads_val = gradient_function([input_file, 0])
    grads_val = grads_val / (np.max(grads_val) + K.epsilon())
    print(grads_val.shape)
    weights = np.mean(grads_val, axis=(1, 2, 3))
    weights.flatten()
    print('weights', weights)
    print('output', output.shape)
    if K.image_data_format() == "channels_last":
        grad_cam = np.ones(output.shape[1:-1], dtype=K.floatx())
    else:
        grad_cam = np.ones(output.shape[2:], dtype=K.floatx())

    for i, w in enumerate(np.transpose(weights)):
        if K.image_data_format() == "channels_last":
            grad_cam += w * output[0, ..., i]
        else:
            grad_cam += w * output[0, i, ...]

    grad_cam = np.maximum(grad_cam, 0)
    print(weights)
    grad_cam = grad_cam
    attMap = np.zeros_like(input_file)

    zoom_factor = [i / (j * 1.0) for i, j in iter(zip(input_file.shape[1:-1], grad_cam.shape))]
    print(input_file.shape[1:-1], grad_cam.shape)
    attMap[..., 0] = zoom(grad_cam, zoom_factor)

    return attMap


def get_saved_foldername(pred_g, threshold, filename):
    if pred_g > threshold:
        return 'G_' + filename[filename.rindex(os.path.sep) + 1:filename.rindex('.')]
    else:
        return 'U_' + filename[filename.rindex(os.path.sep) + 1:filename.rindex('.')]


def get_attMap(cnnModel, filename, activationFunction):
    with open(filename) as f:
        array = np.array([np.fromfile(f, np.uint8).reshape((
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3]))]) / 255
    attMap = GradCAM(activationFunction, array)
    return attMap


def generate_heatmap(filename, attMap, save_to='.', colour_map=plt.cm.rainbow):
    array = np.array([np.fromfile(filename, np.uint8).reshape((
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3]))])
    for i in range(200):
        back = cv2.cvtColor(array[0, i, :, :, 0], cv2.COLOR_GRAY2RGBA) / 255
        colored_image = colour_map(attMap[0, i, :, :, 0])
        weighted = cv2.addWeighted(np.asarray(back[..., :3]), 0.6, np.asarray(colored_image[..., :3]), 0.4, 0)
        final = np.hstack((
            np.flip(weighted, axis=0),
            np.ones((1024, 10, 3)),
            np.flip(np.asarray(back[..., :3]), axis=0)
        ))
        plt.imsave(os.path.join(save_to, 'first_%d.jpg' % i), final)
    for i in range(1024):
        back = cv2.cvtColor(array[0, :, i, :, 0], cv2.COLOR_GRAY2RGBA) / 255
        colored_image = colour_map(attMap[0, :, i, :, 0])
        weighted = cv2.addWeighted(np.asarray(back[..., :3]), 0.6, np.asarray(colored_image[..., :3]), 0.4, 0)
        final = np.hstack((
            weighted,
            np.ones((200, 10, 3)),
            np.asarray(back[..., :3])
        ))
        plt.imsave(os.path.join(save_to, 'second_%d.jpg' % i), final)
    for i in range(200):
        back = cv2.cvtColor(array[0, :, :, i, 0], cv2.COLOR_GRAY2RGBA) / 255
        colored_image = colour_map(attMap[0, :, :, i, 0])
        weighted = cv2.addWeighted(np.asarray(back[..., :3]), 0.6, np.asarray(colored_image[..., :3]), 0.4, 0)
        final = np.hstack((
            np.rot90(weighted),
            np.ones((1024, 10, 3)),
            np.rot90(np.asarray(back[..., :3]))
        ))
        plt.imsave(os.path.join(save_to, 'third_%d.jpg' % i), final)


def get_modified_model(folder, layerIdx=-4, explanation_catagory=1, activation='relu'):
    """
    folder: model folder with a latest file indicates the target model
    layerIdx: The layer index of the last fully convolutional layer after removing the softmax layer
    explanation_catagory: the corresponding encoding for the class-specific activations
    activation: activation function for that layer
    """
    nbClasses = 2

    cnnModel, _ = _load_model(folder, None, weights_only=False)
    print('Model loaded')

    #   Keras functions for getting the GradCAM and guided GradCAM
    activationFunction = prepareGradCAM(cnnModel, explanation_catagory, nbClasses)
    saliency_fn = compileSaliencyFunction(cnnModel, activation, activation_layer=layerIdx)
    print('Model compiling.')
    cnnModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return activationFunction, saliency_fn, cnnModel


if __name__ == '__main__':
    import argparse
    import tensorflow.keras.backend as K
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", type=str, default="./predict_confidence.csv", help='csv path.')
    parser.add_argument("--folder", type=str, default="./trained_model", help='folder path.')
    parser.add_argument("--threshold", type=float, default=0.5, help='threshold for gradable images.')
    args = parser.parse_args()

    threshold = args.threshold

    folder = args.folder
    excel_df = pd.read_csv(args.file)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.allow_soft_placement = True
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras

    # Define a relatively large number as the maximum activation value
    standardize_colour_bar = 50

    for explanation_catagory, excel in [
        (0, excel_df[excel_df['pred_g'] > threshold]),
        (1, excel_df[excel_df['pred_g'] < threshold])
    ]:
        if len(excel) > 0:
            activationFunction, saliency_fn, cnnModel = get_modified_model(
                folder, layerIdx=-4, explanation_catagory=explanation_catagory)
        for idx, row in excel.iterrows():
            filename = os.path.join(row['root'], row['filename'])
            attMap = get_attMap(cnnModel, filename, activationFunction)
            save_to = os.path.join('./GradCAM_outputs', get_saved_foldername(row['pred_g'], threshold, filename))
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            print('====== Processing step %d/%d =======' % (idx + 1, len(excel)), ', saving to', save_to, end='\r')
            generate_heatmap(filename, attMap / standardize_colour_bar, save_to=save_to)
