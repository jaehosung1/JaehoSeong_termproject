# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from multiprocessing import Process, freeze_support
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import imgaug as ia

from ENAS import EfficientNeuralArchitectureSearch
from src.utils import sgdr_learning_rate, get_random_eraser, MixupGenerator
import cv2

num_classes = 10
tfds.disable_progress_bar()
tf.random.set_seed(42)
ia.seed(42)

def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:,:,:,:]])
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = to_categorical(Y_train, num_classes)
    Y_valid = to_categorical(Y_valid, num_classes)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train, Y_train, X_valid, Y_valid

X_train, y_train, X_test, y_test = load_cifar10_data(32, 32)



print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 1
#randaug
def augment(images):
    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())
rand_aug = iaa.RandAugment(n=3, m=7)
train_ds_rand = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
    .map(
        lambda x, y: (tf.image.resize(x, (32, 32)), y),
        num_parallel_calls=AUTO,
    )
    # The returned output of `tf.py_function` contains an unncessary axis of
    # 1-D and we need to remove it.
    .map(
        lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
        num_parallel_calls=AUTO,
    )
    .prefetch(AUTO)
)

def augment(images):
    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())

child_data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(v_l=0, v_h=255))

data_flow_gen = MixupGenerator(
    X_train, y_train, batch_size=128, alpha=0.2, datagen=child_data_gen)()

model_dir = r'C:\Users\5g\PycharmProjects\202206_deeplearning\ENAS-Keras-master\cifar10_weights_h5/cifar_test.h5'
checkpoint = ModelCheckpoint(model_dir, monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='auto')

nt = sgdr_learning_rate(n_Max=0.05, n_min=0.001, ranges=3, init_cycle=10)
reduceLR = ReduceLROnPlateau(
    monitor='val_loss',  # 검증 손실을 기준으로 callback이 호출됩니다
    factor=0.25,          # callback 호출시 학습률을 1/2로 줄입니다
    patience=10,         # epoch 10 동안 개선되지 않으면 callback이 호출됩니다
)

ENAS = EfficientNeuralArchitectureSearch(
    train_ds = train_ds_rand,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    child_network_name="cifar10_cnn",
    child_classes=num_classes,
    child_input_shape=(32, 32, 3),
    num_nodes=6,
    num_opers=7,
    controller_lstm_cell_units=32,
    controller_baseline_decay=0.99,
    controller_opt=Adam(lr=0.00035, decay=1e-3, amsgrad=True),
    controller_batch_size=1,
    controller_epochs=50,
    controller_callbacks=[
        EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    ],
    controller_temperature=5.0,
    controller_tanh_constant=2.5,
    controller_normal_model_file="cifar10_normal_controller.hdf5",
    controller_reduction_model_file="cifar10_reduction_controller.hdf5",
    child_init_filters=32,
    child_network_definition=["N","N", "N","R", "N", "N","N","R","N", "N","N","R" ],
    child_weight_directory="./cifar10_weights",
    child_opt_loss='categorical_crossentropy',
    child_opt=SGD(lr=0.05, nesterov=True),
    child_opt_metrics=['accuracy'],
    child_val_batch_size=128,
    child_batch_size=128,
    child_epochs=len(nt),
    child_lr_scedule=nt,
    start_from_record=True,
    run_on_jupyter=False,
    initialize_child_weight_directory=False,
    save_to_disk=True,
    set_from_dict=True,
    child_callbacks=[checkpoint, early, reduceLR],
    # data_gen=child_data_gen,
    # data_flow_gen=data_flow_gen #edit
)
ENAS.search_neural_architecture()

print(ENAS.best_normal_cell)
print(ENAS.best_reduction_cell)

ENAS.train_best_cells()
