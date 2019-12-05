from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateScheduler,
    CSVLogger,
    TensorBoard
)
import numpy as np
import os

from models.se_resnet_3d import SE_ResNeXt
from models.resnet_3d import Resnet3DBuilder
from utils.data_gen import (
    TrainingSequence,
    TestingSequence,
    PredictSequence
)
from utils.data_processing_3d import DataProcessing3D
from utils.model_utils import save_model, _load_model


def lr_scheduler(initial_lr=1e-2, decay_factor=0.75, step_size=5, min_lr=1e-5):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        if lr > min_lr:
            return lr
        return min_lr

    return LearningRateScheduler(schedule, verbose=1)


if __name__ == '__main__':

    import argparse
    import tensorflow as tf
    import tensorflow.keras.backend as K
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument("--folder", type=str, help='folder path')
    args = parser.parse_args()

    # Choose Targetted GPU Devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.allow_soft_placement = True
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras

    if args.testing:
        mode = 'testing'
    elif args.predict:
        mode = 'predict'
    else:
        mode = 'training'

    target_shape = (200, 1024, 200, 1)
    input_shape = (200, 1024, 200, 1)

    def image_preprocess_fn(img):
        if args.denoise:
            img = DataProcessing3D(img).nl_denoise_3d().image
        return img[:, :, :, :]
    num_outputs = 2

    if args.folder is None:
        from datetime import datetime
        now = datetime.now()  # current date and time
        tag = now.strftime("%Y_%m_%d")

        folder = 'trained_model_%s' % tag

        if os.path.exists(folder):
            model, epoch = _load_model(folder, None, weights_only=False)
        else:
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) == 1:
                device = '/gpu:%s' % os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                device = '/gpu:%s' % os.environ["CUDA_VISIBLE_DEVICES"].split(',')[-1]
            with tf.device(device):
                model = SE_ResNeXt(cardinality=1, blocks=1, depth=32, reduction_ratio=4, init_filters=64,
                                   training=True).build(input_shape=target_shape, num_output=num_outputs, repetitions=5)
                # model = Resnet3DBuilder.build_resnet_18(target_shape, num_outputs, block_fn="basic_block", with_SE=True)
            save_model(folder, model)
            epoch = 0
    else:
        folder = args.folder
        model, epoch = _load_model(folder, None, weights_only=False)
    model.summary()
    print(len(model.layers))

    logger = os.path.join(folder, 'training.log')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    log_append = True if os.path.isfile(logger) else False

    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) == 1:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) - 1

    test_sequence = TestingSequence(
        csv_paths=[
            "./sample_data/datasheets/test_de.csv"
        ],
        batch_size=BATCH_SIZE,
        image_preprocess_fn=image_preprocess_fn,
        shuffle=False,
        target_shape=target_shape,
        input_shape=input_shape
    )

    if mode == 'training':
        train_sequence = TrainingSequence(
            csv_paths=[
                "./sample_data/datasheets/train_de.csv"
            ],
            batch_size=BATCH_SIZE,
            choose_only=10,
            image_preprocess_fn=image_preprocess_fn,
            shuffle=True,
            flip=True,
            shift_range=(10, 10, 10, 0),
            rotation_range=15,
            target_shape=target_shape,
            input_shape=input_shape
        )

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) > 1:
            from tensorflow.python.keras.utils import multi_gpu_model
            model = multi_gpu_model(model, gpus=len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) - 1, cpu_merge=False)

        adam = Adam(lr=0.0, decay=0.1, amsgrad=True)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

        model.fit_generator(
            train_sequence,
            validation_data=test_sequence,
            epochs=100,
            use_multiprocessing=True,
            workers=9,
            callbacks=[
                ModelCheckpoint(
                    '%s/model_{epoch:d}_valacc_{val_acc:.4f}_valoss_{val_loss:.4f}_loss{loss:.4f}_acc_{acc:.4f}.h5' % folder,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=True,
                    mode='auto',
                    period=1),
                CSVLogger(logger, append=log_append),
                lr_scheduler(initial_lr=1e-3, decay_factor=0.75, step_size=3, min_lr=1e-5)
            ],
            max_queue_size=18,
            initial_epoch=epoch
        )
    elif mode == 'testing':
        adam = Adam(lr=1e-3, amsgrad=True)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

        res = model.evaluate_generator(
            test_sequence,
            verbose=1
        )
        print(model.metrics_names)
        print(res)
    elif mode == 'predict':
        import pandas as pd
        df = pd.read_csv("./sample_data/datasheets/predict.csv")
        predict_sequence = PredictSequence(
            dataframe=df,
            batch_size=1,
            shuffle=False,
            image_preprocess_fn=image_preprocess_fn,
            target_shape=(200, 1024, 200, 1),
            input_shape=(200, 1024, 200, 1)
        )
        pred = model.predict_generator(predict_sequence, verbose=1, max_queue_size=0)
        predict_sequence.dataframe['pred_g'] = pred[:, 0]
        predict_sequence.dataframe['pred_u'] = pred[:, 1]
        predict_sequence.dataframe.to_csv('./predict_confidence.csv', index=None)
    else:
        raise ValueError('')
