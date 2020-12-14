#####
# dataset: CIFAR-10 or tinyImageNet

# model: ResNet-34
# ref. paper: Deep Residual Learning for Image Recognition

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import string
import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras import Input, Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)


def block(x, filters, strides=1, shortcut_yn=True, bottleneck_yn=True, name=None):
    """
    Build a residual block

    Arguments:
        x: input tensor
        filters: int, filters of bottleneck layer (3x3 conv layer; "..._2_conv")
        strides: int, default=1, stride of first conv layer (named "..._1_conv")
        shortcut_yn: bool, default=True, use projection shortcut if True, otherwise identity shortcut
        bottleneck_yn: bool, default=True, use bottleneck design (1x1-3x3-1x1) if True, otherwise 3x3-3x3
        name: str, layer and block label

    Returns:
        Output tensor of residual block
    """

    if bottleneck_yn:
        # Save shortcut
        if shortcut_yn:
            # shortcut-residual dimension match (projection shortcut)
            shortcut = Conv2D(filters * 4, 1, strides=strides, name=name + '_shortcut_conv')(x)
            shortcut = BatchNormalization(axis=1, name=name + '_shortcut_bn')(shortcut)
        else:
            # identity shortcut
            shortcut = x

        # 1x1 layer
        conv_1_conv = Conv2D(filters, 1, strides=strides,
                             kernel_initializer='he_normal', name=name + '_1_conv')(x)
        conv_1_bn = BatchNormalization(axis=1, name=name + '_1_bn')(conv_1_conv)
        conv_1_relu = Activation('relu', name=name + '_1_relu')(conv_1_bn)

        # bottleneck (3x3) layer
        conv_2_conv = Conv2D(filters, 3, padding='SAME',
                             kernel_initializer='he_normal', name=name + '_2_conv')(conv_1_relu)
        conv_2_bn = BatchNormalization(axis=1, name=name + '_2_bn')(conv_2_conv)
        conv_2_relu = Activation('relu', name=name + '_2_relu')(conv_2_bn)

        # 1x1 layer
        conv_3_conv = Conv2D(filters * 4, 1,
                             kernel_initializer='he_normal', name=name + '_3_conv')(conv_2_relu)
        conv_3_bn = BatchNormalization(axis=1, name=name + '_3_bn')(conv_3_conv)
        conv_3_relu = Activation('relu', name=name + '_3_relu')(conv_3_bn)

        # Combine shortcut and residual results
        conv_add = tf.keras.layers.Add()([shortcut, conv_3_relu])
        conv_out = Activation('relu', name=name + '_out')(conv_add)

    else:
        if name[4] == '2' or name[-1] != '1':  # for layer < 50, if layer==2 or block!=1, then zero padding
            # Save shortcut
            if shortcut_yn:
                print(x.shape)
                # shortcut-residual dimension match (projection shortcut)
                shortcut = Conv2D(filters, 3, padding='SAME', name=name + '_shortcut_conv')(x)
                shortcut = BatchNormalization(axis=1, name=name + '_shortcut_bn')(shortcut)
                print('shortcut', name, shortcut.shape)
            else:
                # identity shortcut
                shortcut = x
                print('no sc', name)

            print(strides)
            # first 3x3 layer
            conv_1_conv = Conv2D(filters, 3, padding='SAME',
                                 kernel_initializer='he_normal', name=name + '_1_conv')(x)
            conv_1_bn = BatchNormalization(axis=1, name=name + '_1_bn')(conv_1_conv)
            conv_1_relu = Activation('relu', name=name + '_1_relu')(conv_1_bn)
            print(conv_1_relu.shape)

            # second 3x3 layer
            conv_2_conv = Conv2D(filters, 3, padding='SAME',
                                 kernel_initializer='he_normal', name=name + '_2_conv')(conv_1_relu)
            conv_2_bn = BatchNormalization(axis=1, name=name + '_2_bn')(conv_2_conv)
            conv_2_relu = Activation('relu', name=name + '_2_relu')(conv_2_bn)
            print(conv_2_relu.shape)

            # Combine shortcut and residual results
            conv_add = tf.keras.layers.Add()([shortcut, conv_2_relu])
            conv_out = Activation('relu', name=name + '_out')(conv_add)
            print('conv', name)

        else:
            # Save shortcut
            if shortcut_yn:
                print(x.shape)
                # shortcut-residual dimension match (projection shortcut)
                shortcut = Conv2D(filters, 3, strides=1, name=name + '_shortcut_conv')(x)
                shortcut = BatchNormalization(axis=1, name=name + '_shortcut_bn')(shortcut)
                print('shortcut', name, shortcut.shape)
            else:
                # identity shortcut
                shortcut = x
                print('no sc', name)

            print(strides)
            # first 3x3 layer
            conv_1_conv = Conv2D(filters, 3, strides=1,
                                 kernel_initializer='he_normal', name=name + '_1_conv')(x)
            conv_1_bn = BatchNormalization(axis=1, name=name + '_1_bn')(conv_1_conv)
            conv_1_relu = Activation('relu', name=name + '_1_relu')(conv_1_bn)
            print(conv_1_relu.shape)

            # second 3x3 layer
            conv_2_conv = Conv2D(filters, 3, padding='SAME',
                                 kernel_initializer='he_normal', name=name + '_2_conv')(conv_1_relu)
            conv_2_bn = BatchNormalization(axis=1, name=name + '_2_bn')(conv_2_conv)
            conv_2_relu = Activation('relu', name=name + '_2_relu')(conv_2_bn)
            print(conv_2_relu.shape)

            # Combine shortcut and residual results
            conv_add = tf.keras.layers.Add()([shortcut, conv_2_relu])
            conv_out = Activation('relu', name=name + '_out')(conv_add)
            print('conv', name)

    return conv_out


def stacked_block(x, filters, strides=2, blocks=3, bottleneck_yn=True, name=None):
    """
    Stacks a given number of residual blocks

    Arguments:
        x: input tensor
        filters: int, filters of bottleneck layer (3x3 conv layer; "..._2_conv") in a block
        strides: int, default=2, stride of first conv layer (named "..._1_conv") in a block
        blocks: int, default=3, number of blocks in a stack
        bottleneck_yn: bool, default=True, use bottleneck design (1x1-3x3-1x1) if True, otherwise 3x3-3x3 in a block
        name: str, layer label

    Returns:
        Output tensor of a stack of residual blocks
    """

    # Stack blocks
    # Create a projection shortcut in block 1
    # Down-sampling (stride) is performed only in the first block of each layer
    stack = block(x, filters, strides=strides, shortcut_yn=True, bottleneck_yn=bottleneck_yn,
                  name=name + '_block1')
    for b in range(2, blocks + 1):
        # Create an identity shortcut in other blocks
        stack = block(stack, filters, strides=1, shortcut_yn=False, bottleneck_yn=bottleneck_yn,
                      name=name + '_block' + str(b))

    return stack


def resnet(filters, blocks, bottleneck_yn=True, name='resnet'):
    """
    Instantiates the ResNet architecture

    Reference:
    - [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385) (CVPR 2015)

    Arguments:

    Returns:
        A keras.Model instance
    """
    # *** bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    # input
    input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')

    # conv1
    conv1_pad = ZeroPadding2D(padding=3, name='conv1_pad')(input_tensor)
    conv1_conv = Conv2D(64, 7, strides=(2, 2),
                        kernel_initializer='he_normal', name='conv1_conv')(conv1_pad)
    conv1_bn = BatchNormalization(axis=1, name='conv1_bn')(conv1_conv)
    conv1_relu = Activation('relu', name='conv1_relu')(conv1_bn)

    # conv2_max pooling
    pool1_pad = ZeroPadding2D(padding=1, name='pool1_pad')(conv1_relu)
    pool1_pool = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(pool1_pad)

    # conv layer 2-5
    conv2 = stacked_block(pool1_pool, filters=filters[0], strides=1, blocks=blocks[0], bottleneck_yn=bottleneck_yn,
                          name='conv2')
    conv3 = stacked_block(conv2, filters=filters[1], strides=2, blocks=blocks[1], bottleneck_yn=bottleneck_yn,
                          name='conv3')
    conv4 = stacked_block(conv3, filters=filters[2], strides=2, blocks=blocks[2], bottleneck_yn=bottleneck_yn,
                          name='conv4')
    conv5 = stacked_block(conv4, filters=filters[3], strides=2, blocks=blocks[3], bottleneck_yn=bottleneck_yn,
                          name='conv5')

    # FC
    avgpool = GlobalAveragePooling2D(name='avgpool')(conv5)
    output_tensor = Dense(10, activation='softmax', name='softmax')(avgpool)

    # Create a model
    model = Model(input_tensor, output_tensor, name=name)

    return model


if __name__ == '__main__':
    # Set the value of hyper-parameters
    learning_rate = 0.0001
    epochs = 20
    batch_size = 16

    # Results by hyper-parameters
    # ==> learning rate: 0.0001; epoch: 20; batch size: 16; loss: 1.1171 - accuracy: 0.7277

    # Load cifar10 dataset
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the dataset
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Create a model
    resnet34 = resnet(filters=[64, 128, 256, 512], blocks=[3, 4, 6, 3], bottleneck_yn=False, name='resnet34')
    # resnet50 = resnet(filters=[64, 128, 256, 512], blocks=[3, 4, 6, 3], bottleneck_yn=True, name='resnet34')
    resnet34.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    resnet34.compile(loss='sparse_categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    # Train the model to adjust parameters to minimize the loss
    resnet34.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Test the model with test set
    resnet34.evaluate(x_test, y_test, verbose=2)
