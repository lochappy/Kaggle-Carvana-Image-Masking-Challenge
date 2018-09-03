from keras.models import Model
from keras.layers import * #Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Flatten, Lambda
from keras.optimizers import RMSprop
from keras import regularizers

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up1 = Conv2D(num_classes, (1, 1))(up1)
    classify0 = Activation('sigmoid',name="mask")(up1)
    classify1 = Lambda(lambda x: x*255,name="mse")(classify0)
    #classify1 = Flatten(name="mae")(classify0)

    model = Model(inputs=inputs, outputs=[classify0,classify1])

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_256(input_shape=(256, 256, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0 = Conv2D(num_classes, (1, 1))(up0)
    classify0 = Activation('sigmoid',name="mask")(up0)
    classify1 = Lambda(lambda x: x*255,name="mse")(classify0)
    #classify1 = Flatten(name="mae")(classify0)

    model = Model(inputs=inputs, outputs=[classify0,classify1])

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_512(input_shape=(512, 512, 3),
                 num_classes=1, autoencoder = False):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0a = Conv2D(num_classes, (1, 1))(up0a)
    classify0 = Activation('sigmoid',name="mask")(up0a)
    classify1 = Lambda(lambda x: x*255,name="mse")(classify0)
    #classify1 = Flatten(name="mae")(classify0)

    model = Model(inputs=inputs, outputs=[classify0,classify1])

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1, autoencoder = False):
    inputs = Input(shape=input_shape, name="input_img")
    
    initializer = "he_uniform" #'he_normal'
    regularizer = lambda: regularizers.l1_l2(l1=0.0001,l2=0.0001)
    #regularizer = lambda: None
    
    if autoencoder:
        prefix = 'audec'
    else:
        prefix = 'dec'
    
    #enc1 1024
    down0b = Conv2D(8, (3, 3), padding='same', name="enc0_conv1",kernel_initializer=initializer, kernel_regularizer=regularizer())(inputs)
    down0b = BatchNormalization(name="enc0_bn1")(down0b)
    down0b = Activation('relu', name="enc0_relu1")(down0b)
    
    down0b = Conv2D(8, (3, 3), padding='same', name="enc0_conv2",kernel_initializer=initializer, kernel_regularizer=regularizer())(down0b)
    down0b = BatchNormalization(name="enc0_bn2")(down0b)
    down0b = Activation('relu', name="enc0_relu2")(down0b)
    
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2), name="enc0_pool")(down0b)

    #enc2 512
    down0a = Conv2D(16, (3, 3), padding='same', name="enc1_conv1",kernel_initializer=initializer, kernel_regularizer=regularizer())(down0b_pool)
    down0a = BatchNormalization(name="enc1_bn1")(down0a)
    down0a = Activation('relu', name="enc1_relu1")(down0a)
    down0a = Conv2D(16, (3, 3), padding='same', name="enc1_conv2",kernel_initializer=initializer, kernel_regularizer=regularizer())(down0a)
    down0a = BatchNormalization(name="enc1_bn2")(down0a)
    down0a = Activation('relu', name="enc1_relu2")(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2), name="enc1_pool")(down0a)
    
    # enc 256
    down0 = Conv2D(32, (3, 3), padding='same', name='enc2_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down0a_pool)
    down0 = BatchNormalization(name='enc2_bn1')(down0)
    down0 = Activation('relu', name='enc2_relu1')(down0)
    down0 = Conv2D(32, (3, 3), padding='same', name='enc2_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(down0)
    down0 = BatchNormalization(name='enc2_bn2')(down0)
    down0 = Activation('relu', name='enc2_relu2')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2), name='enc2_pool')(down0)
    
    #enc3 128
    down1 = Conv2D(64, (3, 3), padding='same', name='enc3_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down0_pool)
    down1 = BatchNormalization(name='enc3_bn1')(down1)
    down1 = Activation('relu', name='enc3_relu1')(down1)
    down1 = Conv2D(64, (3, 3), padding='same', name='enc3_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(down1)
    down1 = BatchNormalization(name='enc3_bn2')(down1)
    down1 = Activation('relu', name='enc3_relu2')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='enc3_pool')(down1)
    
    #enc4 64
    down2 = Conv2D(128, (3, 3), padding='same', name='enc4_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down1_pool)
    down2 = BatchNormalization(name='enc4_bn1')(down2)
    down2 = Activation('relu', name='enc4_relu1')(down2)
    down2 = Conv2D(128, (3, 3), padding='same', name='enc4_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(down2)
    down2 = BatchNormalization(name='enc4_bn2')(down2)
    down2 = Activation('relu',name='enc4_relu2')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2),name='enc4_pool')(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same',name='enc5_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down2_pool)
    down3 = BatchNormalization(name='enc5_bn1')(down3)
    down3 = Activation('relu',name='enc5_relu1')(down3)
    down3 = Conv2D(256, (3, 3), padding='same',name='enc5_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(down3)
    down3 = BatchNormalization(name='enc5_bn2')(down3)
    down3 = Activation('relu',name='enc5_relu2')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2),name='enc5_pool')(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same',name='enc6_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down3_pool)
    down4 = BatchNormalization(name='enc6_bn1')(down4)
    down4 = Activation('relu',name='enc6_relu1')(down4)
    down4 = Conv2D(512, (3, 3), padding='same',name='enc6_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(down4)
    down4 = BatchNormalization(name='enc6_bn2')(down4)
    down4 = Activation('relu',name='enc6_relu2')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2),name='enc6_pool')(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same',name='enc7_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(down4_pool)
    center = BatchNormalization(name='enc7_bn1')(center)
    center = Activation('relu',name='enc7_relu1')(center)
    center = Conv2D(1024, (3, 3), padding='same',name='enc7_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(center)
    center = BatchNormalization(name='enc7_bn2')(center)
    center = Activation('relu',name='enc7_relu2')(center)
    # center

    up4 = UpSampling2D((2, 2),name=prefix + '7_up')(center)
    if not autoencoder:
        up4 = concatenate([down4, up4], axis=3,name=prefix + '7_concat')
        up4 = Conv2D(512, (3, 3), padding='same',name=prefix + '7_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up4)
        up4 = BatchNormalization(name=prefix + '7_bn1')(up4)
        up4 = Activation('relu',name=prefix + '7_relu1')(up4)
    up4 = Conv2D(512, (3, 3), padding='same', name=prefix + '7_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up4)
    up4 = BatchNormalization(name=prefix + '7_bn2')(up4)
    up4 = Activation('relu',name=prefix + '7_relu2')(up4)
    up4 = Conv2D(512, (3, 3), padding='same',name=prefix + '7_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up4)
    up4 = BatchNormalization(name=prefix + '7_bn3')(up4)
    up4 = Activation('relu',name=prefix + '7_relu3')(up4)
    # 16

    up3 = UpSampling2D((2, 2),name=prefix + '6_up')(up4)
    if not autoencoder:
        up3 = concatenate([down3, up3], axis=3,name=prefix + '6_concat')
        up3 = Conv2D(256, (3, 3), padding='same', name=prefix + '6_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up3)
        up3 = BatchNormalization(name=prefix + '6_bn1')(up3)
        up3 = Activation('relu', name=prefix + '6_relu1')(up3)
    up3 = Conv2D(256, (3, 3), padding='same', name=prefix + '6_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up3)
    up3 = BatchNormalization(name=prefix + '6_bn2')(up3)
    up3 = Activation('relu', name=prefix + '6_relu2')(up3)
    up3 = Conv2D(256, (3, 3), padding='same', name=prefix + '6_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up3)
    up3 = BatchNormalization(name=prefix + '6_bn3')(up3)
    up3 = Activation('relu', name=prefix + '6_relu3')(up3)
    # 32

    up2 = UpSampling2D((2, 2),name=prefix + '5_up')(up3)
    if not autoencoder:
        up2 = concatenate([down2, up2], axis=3,name=prefix + '5_concat')
        up2 = Conv2D(128, (3, 3), padding='same',name=prefix + '5_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up2)
        up2 = BatchNormalization(name=prefix + '5_bn1')(up2)
        up2 = Activation('relu',name=prefix + '5_relu1')(up2)
    up2 = Conv2D(128, (3, 3), padding='same',name=prefix + '5_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up2)
    up2 = BatchNormalization(name=prefix + '5_bn2')(up2)
    up2 = Activation('relu',name=prefix + '5_relu2')(up2)
    up2 = Conv2D(128, (3, 3), padding='same',name=prefix + '5_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up2)
    up2 = BatchNormalization(name=prefix + '5_bn3')(up2)
    up2 = Activation('relu',name=prefix + '5_relu3')(up2)
    # 64

    up1 = UpSampling2D((2, 2),name=prefix + '4_up')(up2)
    if not autoencoder:
        up1 = concatenate([down1, up1], axis=3,name=prefix + '4_concat')
        up1 = Conv2D(64, (3, 3), padding='same',name=prefix + '4_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up1)
        up1 = BatchNormalization(name=prefix + '4_bn1')(up1)
        up1 = Activation('relu',name=prefix + '4_relu1')(up1)
    up1 = Conv2D(64, (3, 3), padding='same',name=prefix + '4_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up1)
    up1 = BatchNormalization(name=prefix + '4_bn2')(up1)
    up1 = Activation('relu',name=prefix + '4_relu2')(up1)
    up1 = Conv2D(64, (3, 3), padding='same',name=prefix + '4_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up1)
    up1 = BatchNormalization(name=prefix + '4_bn3')(up1)
    up1 = Activation('relu',name=prefix + '4_relu3')(up1)
    # 128

    up0 = UpSampling2D((2, 2),name=prefix + '3_up')(up1)
    if not autoencoder:
        up0 = concatenate([down0, up0], axis=3,name=prefix + '3_concat')
        up0 = Conv2D(32, (3, 3), padding='same',name=prefix + '3_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0)
        up0 = BatchNormalization(name=prefix + '3_bn1')(up0)
        up0 = Activation('relu',name=prefix + '3_relu1')(up0)
    up0 = Conv2D(32, (3, 3), padding='same',name=prefix + '3_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0)
    up0 = BatchNormalization(name=prefix + '3_bn2')(up0)
    up0 = Activation('relu',name=prefix + '3_relu2')(up0)
    up0 = Conv2D(32, (3, 3), padding='same',name=prefix + '3_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0)
    up0 = BatchNormalization(name=prefix + '3_bn3')(up0)
    up0 = Activation('relu',name=prefix + '3_relu3')(up0)
    # 256

    up0a = UpSampling2D((2, 2),name=prefix + '2_up')(up0)
    if not autoencoder:
        up0a = concatenate([down0a, up0a], axis=3,name=prefix + '2_concat')
        up0a = Conv2D(16, (3, 3), padding='same',name=prefix + '2_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0a)
        up0a = BatchNormalization(name=prefix + '2_bn1')(up0a)
        up0a = Activation('relu',name=prefix + '2_relu1')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same',name=prefix + '2_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0a)
    up0a = BatchNormalization(name=prefix + '2_bn2')(up0a)
    up0a = Activation('relu',name=prefix + '2_relu2')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same',name=prefix + '2_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0a)
    up0a = BatchNormalization(name=prefix + '2_bn3')(up0a)
    up0a = Activation('relu',name=prefix + '2_relu3')(up0a)
    # 512

    up0b = UpSampling2D((2, 2),name=prefix + '1_up')(up0a)
    if not autoencoder:
        up0b = concatenate([down0b, up0b], axis=3,name=prefix + '1_concat')
        up0b = Conv2D(8, (3, 3), padding='same',name=prefix + '1_conv1',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0b)
        up0b = BatchNormalization(name=prefix + '1_bn1')(up0b)
        up0b = Activation('relu',name=prefix + '1_relu1')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same',name=prefix + '1_conv2',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0b)
    up0b = BatchNormalization(name=prefix + '1_bn2')(up0b)
    up0b = Activation('relu',name=prefix + '1_relu2')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same',name=prefix + '1_conv3',kernel_initializer=initializer, kernel_regularizer=regularizer())(up0b)
    up0b = BatchNormalization(name=prefix + '1_bn3')(up0b)
    up0b = Activation('relu',name=prefix + '1_relu3')(up0b)
    # 1024

    if autoencoder:
        up0b = Conv2D(3, (1, 1),activation='sigmoid', name='reconstruction',kernel_initializer=initializer)(up0b)
        up0b = Lambda(lambda x: x*255,name="scale_up")(up0b)
        model = Model(inputs=inputs, outputs=up0b)
    else:
        classify0 = Conv2D(num_classes, (1, 1),activation='sigmoid', name = 'mask')(up0b)
        classify1 = Lambda(lambda x: x*255,name="mse")(classify0)

        model = Model(inputs=inputs, outputs=[classify0,classify1])

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
