#####
# dataset: CIFAR-10

# model: ResNet-50
# ref. paper: Deep Residual Learning for Image Recognition

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras import Input, Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Load cifar10 dataset
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
# batch size = 64 or 32 does not work (running out of memory)
# ==> learning rate: 0.001; Epoch: 20; batch size: 16;
# ==> learning rate: 0.0001; Epoch: 20; batch size: 16; loss: 1.4228 - accuracy: 0.7004
# ==> learning rate: 0.0005; Epoch: 20; batch size: 16;

# Initiate a ResNet50 architecture
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')  # 32,32,3

# conv1

# kernel_initializer: a statistical distribution or function to use for initialising the weights
#   - glorot_uniform: (default) also called Xavier uniform;
#       Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))
#       (fan_in is the number of input units in the weight tensor and fan_out is the number of output units).
#   - he_normal:
#       Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)

x = ZeroPadding2D(padding=3, name='conv1_pad')(input_tensor)
x = Conv2D(64, 7, strides=(2, 2),  # stride -> downsampling
           kernel_initializer='he_normal', name='conv1')(x)
x = BatchNormalization(axis=1, name='conv1_bn')(x)  # In tf, batch channel comes first ==> axis=1
x = Activation('relu', name='conv1_relu')(x)  # 16,16,64

# conv2_max pooling
x = ZeroPadding2D(padding=1, name='conv2_maxpool_pad')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), name='conv2_maxpool')(x)  # 8,8,64

# conv2_a
shortcut = Conv2D(256, 1, name='conv2_shortcut')(x)  # shortcut-residual dimension match (projection shortcut)
shortcut = BatchNormalization(axis=1, name='conv2-shortcut_bn')(shortcut)

x = Conv2D(64, 1,
           kernel_initializer='he_normal', name='conv2a_1')(x)
x = BatchNormalization(axis=1, name='conv2a_1bn')(x)
x = Activation('relu', name='conv2a_1relu')(x)

x = Conv2D(64, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv2a_2')(x)
x = BatchNormalization(axis=1, name='conv2a_2bn')(x)
x = Activation('relu', name='conv2a_2relu')(x)

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv2a_3')(x)
x = BatchNormalization(axis=1, name='conv2a_3bn')(x)
x = Activation('relu', name='conv2a_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])  # element-wise computation
x = Activation('relu', name='conv2a_out')(x)  # 8, 8, 256

# conv2_b
shortcut = x

x = Conv2D(64, 1,
           kernel_initializer='he_normal', name='conv2b_1')(x)
x = BatchNormalization(axis=1, name='conv2b_1bn')(x)
x = Activation('relu', name='conv2b_1relu')(x)

x = Conv2D(64, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv2b_2')(x)
x = BatchNormalization(axis=1, name='conv2b_2bn')(x)
x = Activation('relu', name='conv2b_2relu')(x)

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv2b_3')(x)
x = BatchNormalization(axis=1, name='conv2b_3bn')(x)
x = Activation('relu', name='conv2b_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv2b_out')(x)

# conv2_c
shortcut = x

x = Conv2D(64, 1,
           kernel_initializer='he_normal', name='conv2c_1')(x)
x = BatchNormalization(axis=1, name='conv2c_1bn')(x)
x = Activation('relu', name='conv2c_1relu')(x)

x = Conv2D(64, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv2c_2')(x)
x = BatchNormalization(axis=1, name='conv2c_2bn')(x)
x = Activation('relu', name='conv2c_2relu')(x)

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv2c_3')(x)
x = BatchNormalization(axis=1, name='conv2c_3bn')(x)
x = Activation('relu', name='conv2c_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv2c_out')(x)

# conv3_a
shortcut = Conv2D(512, 1, strides=(2, 2),  # Down-sampling is performed by conv3_a, conv4_a, and conv5_a with stride=2
                  name='conv3_shortcut')(x)
shortcut = BatchNormalization(axis=1, name='conv3_shortcut_bn')(shortcut)

x = Conv2D(128, 1, strides=(2, 2),
           kernel_initializer='he_normal', name='conv3a_1')(x)
x = BatchNormalization(axis=1, name='conv3a_1bn')(x)
x = Activation('relu', name='conv3a_1relu')(x)

x = Conv2D(128, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv3a_2')(x)
x = BatchNormalization(axis=1, name='conv3a_2bn')(x)
x = Activation('relu', name='conv3a_2relu')(x)

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv3a_3')(x)
x = BatchNormalization(axis=1, name='conv3a_3bn')(x)
x = Activation('relu', name='conv3a_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv3a_out')(x)  # 8, 8, 512

# conv3_b
shortcut = x

x = Conv2D(128, 1,
           kernel_initializer='he_normal', name='conv3b_1')(x)
x = BatchNormalization(axis=1, name='conv3b_1bn')(x)
x = Activation('relu', name='conv3b_1relu')(x)

x = Conv2D(128, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv3b_2')(x)
x = BatchNormalization(axis=1, name='conv3b_2bn')(x)
x = Activation('relu', name='conv3b_2relu')(x)

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv3b_3')(x)
x = BatchNormalization(axis=1, name='conv3b_3bn')(x)
x = Activation('relu', name='conv3b_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv3b_out')(x)

# conv3_c
shortcut = x

x = Conv2D(128, 1,
           kernel_initializer='he_normal', name='conv3c_1')(x)
x = BatchNormalization(axis=1, name='conv3c_1bn')(x)
x = Activation('relu', name='conv3c_1relu')(x)

x = Conv2D(128, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv3c_2')(x)
x = BatchNormalization(axis=1, name='conv3c_2bn')(x)
x = Activation('relu', name='conv3c_2relu')(x)

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv3c_3')(x)
x = BatchNormalization(axis=1, name='conv3c_3bn')(x)
x = Activation('relu', name='conv3c_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv3c_out')(x)

# conv3_d
shortcut = x

x = Conv2D(128, 1,
           kernel_initializer='he_normal', name='conv3d_1')(x)
x = BatchNormalization(axis=1, name='conv3d_1bn')(x)
x = Activation('relu', name='conv3d_1relu')(x)

x = Conv2D(128, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv3d_2')(x)
x = BatchNormalization(axis=1, name='conv3d_2bn')(x)
x = Activation('relu', name='conv3d_2relu')(x)

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv3d_3')(x)
x = BatchNormalization(axis=1, name='conv3d_3bn')(x)
x = Activation('relu', name='conv3d_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv3d_out')(x)

# conv4_a
shortcut = Conv2D(1024, 1, strides=(2, 2), name='conv4_shortcut')(x)
shortcut = BatchNormalization(axis=1, name='conv4_shortcut_bn')(shortcut)

x = Conv2D(256, 1, strides=(2, 2),
           kernel_initializer='he_normal', name='conv4a_1')(x)
x = BatchNormalization(axis=1, name='conv4a_1bn')(x)
x = Activation('relu', name='conv4a_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4a_2')(x)
x = BatchNormalization(axis=1, name='conv4a_2bn')(x)
x = Activation('relu', name='conv4a_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4a_3')(x)
x = BatchNormalization(axis=1, name='conv4a_3bn')(x)
x = Activation('relu', name='conv4a_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4a_out')(x)

# conv4_b
shortcut = x

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv4b_1')(x)
x = BatchNormalization(axis=1, name='conv4b_1bn')(x)
x = Activation('relu', name='conv4b_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4b_2')(x)
x = BatchNormalization(axis=1, name='conv4b_2bn')(x)
x = Activation('relu', name='conv4b_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4b_3')(x)
x = BatchNormalization(axis=1, name='conv4b_3bn')(x)
x = Activation('relu', name='conv4b_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4b_out')(x)

# conv4_c
shortcut = x

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv4c_1')(x)
x = BatchNormalization(axis=1, name='conv4c_1bn')(x)
x = Activation('relu', name='conv4c_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4c_2')(x)
x = BatchNormalization(axis=1, name='conv4c_2bn')(x)
x = Activation('relu', name='conv4c_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4c_3')(x)
x = BatchNormalization(axis=1, name='conv4c_3bn')(x)
x = Activation('relu', name='conv4c_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4c_out')(x)

# conv4_d
shortcut = x

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv4d_1')(x)
x = BatchNormalization(axis=1, name='conv4d_1bn')(x)
x = Activation('relu', name='conv4d_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4d_2')(x)
x = BatchNormalization(axis=1, name='conv4d_2bn')(x)
x = Activation('relu', name='conv4d_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4d_3')(x)
x = BatchNormalization(axis=1, name='conv4d_3bn')(x)
x = Activation('relu', name='conv4d_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4d_out')(x)

# conv4_e
shortcut = x

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv4e_1')(x)
x = BatchNormalization(axis=1, name='conv4e_1bn')(x)
x = Activation('relu', name='conv4e_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4e_2')(x)
x = BatchNormalization(axis=1, name='conv4e_2bn')(x)
x = Activation('relu', name='conv4e_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4e_3')(x)
x = BatchNormalization(axis=1, name='conv4e_3bn')(x)
x = Activation('relu', name='conv4e_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4e_out')(x)

# conv4_f
shortcut = x

x = Conv2D(256, 1,
           kernel_initializer='he_normal', name='conv4f_1')(x)
x = BatchNormalization(axis=1, name='conv4f_1bn')(x)
x = Activation('relu', name='conv4f_1relu')(x)

x = Conv2D(256, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv4f_2')(x)
x = BatchNormalization(axis=1, name='conv4f_2bn')(x)
x = Activation('relu', name='conv4f_2relu')(x)

x = Conv2D(1024, 1,
           kernel_initializer='he_normal', name='conv4f_3')(x)
x = BatchNormalization(axis=1, name='conv4f_3bn')(x)
x = Activation('relu', name='conv4f_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv4f_out')(x)

# conv5_a
shortcut = Conv2D(2048, 1, strides=(2, 2), name='conv5_shortcut')(x)
shortcut = BatchNormalization(axis=1, name='conv5_shortcut_bn')(shortcut)

x = Conv2D(512, 1, strides=(2, 2),
           kernel_initializer='he_normal', name='conv5a_1')(x)
x = BatchNormalization(axis=1, name='conv5a_1bn')(x)
x = Activation('relu', name='conv5a_1relu')(x)

x = Conv2D(512, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv5a_2')(x)
x = BatchNormalization(axis=1, name='conv5a_2bn')(x)
x = Activation('relu', name='conv5a_2relu')(x)

x = Conv2D(2048, 1,
           kernel_initializer='he_normal', name='conv5a_3')(x)
x = BatchNormalization(axis=1, name='conv5a_3bn')(x)
x = Activation('relu', name='conv5a_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv5a_out')(x)

# conv5_b
shortcut = x

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv5b_1')(x)
x = BatchNormalization(axis=1, name='conv5b_1bn')(x)
x = Activation('relu', name='conv5b_1relu')(x)

x = Conv2D(512, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv5b_2')(x)
x = BatchNormalization(axis=1, name='conv5b_2bn')(x)
x = Activation('relu', name='conv5b_2relu')(x)

x = Conv2D(2048, 1,
           kernel_initializer='he_normal', name='conv5b_3')(x)
x = BatchNormalization(axis=1, name='conv5b_3bn')(x)
x = Activation('relu', name='conv5b_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv5b_out')(x)

# conv5_c
shortcut = x

x = Conv2D(512, 1,
           kernel_initializer='he_normal', name='conv5c_1')(x)
x = BatchNormalization(axis=1, name='conv5c_1bn')(x)
x = Activation('relu', name='conv5c_1relu')(x)

x = Conv2D(512, 3, padding='SAME',
           kernel_initializer='he_normal', name='conv5c_2')(x)
x = BatchNormalization(axis=1, name='conv5c_2bn')(x)
x = Activation('relu', name='conv5c_2relu')(x)

x = Conv2D(2048, 1,
           kernel_initializer='he_normal', name='conv5c_3')(x)
x = BatchNormalization(axis=1, name='conv5c_3bn')(x)
x = Activation('relu', name='conv5c_3relu')(x)

x = tf.keras.layers.Add()([shortcut, x])
x = Activation('relu', name='conv5c_out')(x)

# FC
x = GlobalAveragePooling2D(name='avgpooling')(x)
output_tensor = Dense(10, activation='softmax', name='softmax')(x)

# Create a model
resnet50 = Model(input_tensor, output_tensor, name='resnet50')
resnet50.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
resnet50.compile(loss='sparse_categorical_crossentropy',
                 optimizer=opt,
                 metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
resnet50.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Test the model with test set
resnet50.evaluate(x_test, y_test, verbose=2)


'''
# Build comparable vgg16 model with existing module
resnet50_module = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling='max',
    classes=10,
    classifier_activation='softmax'
)
resnet50_module.summary()

resnet50_module.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
resnet50_module.fit(x_train, y_train, epochs=1)

# Test the model with test set
resnet50_module.evaluate(x_test, y_test, verbose=2)
# ==> learning rate: 0.001 (default); epoch: 1; loss: 3.0760 - accuracy: 0.4132
# ==> learning rate: 0.001 (default); epoch: 20; 
'''