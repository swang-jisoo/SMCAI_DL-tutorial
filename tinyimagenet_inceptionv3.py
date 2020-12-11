#####
# dataset: tinyImageNet

# model: inception v3
# ref.
#   - Going deeper with convolutions (inception v1)
#   - Rethinking the Inception Architecture for Computer Vision (inception v2, v3)

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# General problem of Deep Neural Network
# ==> over-fitting & computationally expensive due to a large number of parameters
# Suggested solutions
# 1. Sparsely connected network / layer-by-layer (layer-wise) construction
#   - Opposed to fully connected network
#   - Cluster neurons with highly correlated outputs of the previous activation layer (Hebbian principle)
#   ==> Compared to dense matrix calculation (parallel computing), sparse matrix computation is less efficient
# 2. Inception architecture
#   - Intermediate step: sparsity at filter level + dense matrix computation
#   - The idea of sparsity is corresponded to a variation in the location of information in the image
#       ==> use a different size of kernels (larger one for global info; smaller for local)
#       ==> naive: [1x1 (cross-channel correlation)] + [3x3 + 5x5 (spatial correlation)] + 3x3maxpool
#   - To reduce computational cost:
#       ==> inception-v1: 1x1 + 1x1-3x3 + 1x1-5x5 + 3x3maxpool-1x1 -> filter concatenation
#       * 1x1 conv benefits: increased representational power of NN + dimension reduction (network-in-network)
#       ==> inception-v2,v3: 1x1 + 1x1-3x3 + 1x1-3x3-3x3 + 3x3maxpool-1x1 -> filter concatenation (factorization)

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras import Input, Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Load tiny imangenet dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Set the value of hyper-parameters
learning_rate = 0.0005
epochs = 20
batch_size = 16

# Results by hyper-parameters
# ==>

# Initiate a ResNet50 architecture
# input at least 128x128 ==> resize (upsampling)
input_tensor = Input(shape=(75, 75, 3), dtype='float32', name='input')  # 32,32,3

# conv1




'''
# tf.keras module
incepv3_module = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(75, 75, 3),
    pooling='max',
    classes=10,
    classifier_activation="softmax",
)

incepv3_module.summary()

incepv3_module.compile(optimizer='rmsprop',
                       loss={'predictions': 'categorical_crossentropy',
                             'aux_classifier': 'categorical_crossentropy'},
                       loss_weights={'predictions': 1., 'aux_classifier': 0.2},
                       metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
# NOTE: Need to resize the dataset
incepv3_module.fit(x_train, {'predictions': y_train, 'aux_classifier': y_train},
                   batch_size=16, epochs=1)

# Test the model with test set
incepv3_module.evaluate(x_test, y_test, verbose=2)
# ==> learning rate: 0.001 (default); epoch: 1;
# ==> learning rate: 0.001 (default); epoch: 20;
'''
