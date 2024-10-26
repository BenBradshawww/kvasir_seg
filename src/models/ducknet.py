import tensorflow as tf

def conv_block(x:tf.Tensor, filters:int, dilation_rate:int=1, strides:int=1):

    x = tf.keras.layers.Conv2D(filters=filters,
                                kernel_size=3,
                                strides=strides,
                                padding='same',
                                use_bias='False',
                                dilation_rate=dilation_rate)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    return x



def residual_block(x:tf.Tensor, filters:int):

    skip = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=1,
                                  strides=1,
                                  padding='same',
                                  use_bias='False')(x)

    x = conv_block(x=x, filters=filters, dilation_rate=1, strides=1)
    
    x = conv_block(x=x, filters=filters, dilation_rate=1, strides=1)

    x = tf.keras.layers.Add()([x, skip])

    return x

def midscope_block(x:tf.Tensor, filters:int):

    x = conv_block(x=x, filters=filters, dilation_rate=1, strides=1)
    
    x = conv_block(x=x, filters=filters, dilation_rate=2, strides=1)

    return x


def widescope_block(x:tf.Tensor, filters:int):

    x = conv_block(x=x, filters=filters, dilation_rate=1, strides=1)
    
    x = conv_block(x=x, filters=filters, dilation_rate=2, strides=1)

    x = conv_block(x=x, filters=filters, dilation_rate=3, strides=1)

    return x

def seperable_block(x:tf.Tensor, filters:int, size:int):

    x = x = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=(1, size),
                                    strides=1,
                                    padding='same',
                                    use_bias='False')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = x = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=(size, 1),
                                    strides=1,
                                    padding='same',
                                    use_bias='False')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    return x

def duck_block(x:tf.Tensor, filters:int):

    midscope_output = midscope_block(x=x, filters=filters)

    widescope_output = widescope_block(x=x, filters=filters)

    seperable_output = seperable_block(x=x, filters=filters, size=3)

    residual_output_1 = residual_block(x=x, filters=filters)

    residual_output_2 = residual_block(x=x, filters=filters)
    residual_output_2 = residual_block(x=residual_output_2, filters=filters)

    residual_output_3 = residual_block(x=x, filters=filters)
    residual_output_3 = residual_block(x=residual_output_3, filters=filters)
    residual_output_3 = residual_block(x=residual_output_3, filters=filters)

    x = tf.keras.layers.Add()([midscope_output, widescope_output, seperable_output, residual_output_1, residual_output_2, residual_output_3])

    x = tf.keras.layers.BatchNormalization()(x)

    return x

def ducknet(input_shape:tuple[int], start_filters:int):

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder blocks
    e1 = conv_block(x=inputs, filters=start_filters*2, dilation_rate=1, strides=2)
    e2 = conv_block(x=e1, filters=start_filters*4, dilation_rate=1, strides=2)
    e3 = conv_block(x=e2, filters=start_filters*8, dilation_rate=1, strides=2)
    e4 = conv_block(x=e3, filters=start_filters*16, dilation_rate=1, strides=2)
    e5 = conv_block(x=e4, filters=start_filters*32, dilation_rate=1, strides=2)

    # Duck encoder block
    duck_e0 = duck_block(x=inputs, filters=start_filters)

    duck_e1 = conv_block(x=duck_e0, filters=start_filters*2, strides=2)
    duck_e1 = tf.keras.layers.Add()([e1, duck_e1])
    duck_e1 = duck_block(x=duck_e1, filters=start_filters*2)

    duck_e2 = conv_block(x=duck_e1, filters=start_filters*4, strides=2)
    duck_e2 = tf.keras.layers.Add()([e2, duck_e2])
    duck_e2 = duck_block(x=duck_e2, filters=start_filters*4)

    duck_e3 = conv_block(x=duck_e2, filters=start_filters*8, strides=2)
    duck_e3 = tf.keras.layers.Add()([e3, duck_e3])
    duck_e3 = duck_block(x=duck_e3, filters=start_filters*8)

    duck_e4 = conv_block(x=duck_e3, filters=start_filters*16, strides=2)
    duck_e4 = tf.keras.layers.Add()([e4, duck_e4])
    duck_e4 = duck_block(x=duck_e4, filters=start_filters*16)

    # Bridge blocks
    duck_e5 = conv_block(x=duck_e4, filters=start_filters*32, strides=2)
    duck_e5 = tf.keras.layers.Add()([e5, duck_e5])
    duck_e5 = residual_block(x=duck_e5, filters=start_filters*32)
    duck_e5 = residual_block(x=duck_e5, filters=start_filters*16)

    # Duck decoder blocks
    duck_d4 = tf.keras.layers.UpSampling2D(size=2)(duck_e5)
    duck_d4 = tf.keras.layers.Add()([e4, duck_d4])
    duck_d4 = duck_block(x=duck_d4, filters=start_filters*8)

    duck_d3 = tf.keras.layers.UpSampling2D(size=2)(duck_d4)
    duck_d3 = tf.keras.layers.Add()([e3, duck_d3])
    duck_d3 = duck_block(x=duck_d3, filters=start_filters*4)

    duck_d2 = tf.keras.layers.UpSampling2D(size=2)(duck_d3)
    duck_d2 = tf.keras.layers.Add()([e2, duck_d2])
    duck_d2 = duck_block(x=duck_d2, filters=start_filters*2)

    duck_d1 = tf.keras.layers.UpSampling2D(size=2)(duck_d2)
    duck_d1 = tf.keras.layers.Add()([e1, duck_d1])
    duck_d1 = duck_block(x=duck_d1, filters=start_filters*1)

    duck_d0 = tf.keras.layers.UpSampling2D(size=2)(duck_d1)
    duck_d0 = tf.keras.layers.Add()([duck_e0, duck_d0])
    duck_d0 = duck_block(x=duck_d0, filters=start_filters)


    # 1x1 convolutions
    outputs = tf.keras.layers.Conv2D(filters=1,
                                        kernel_size=1,
                                        strides=1,
                                        padding="same",
                                        activation='sigmoid')(duck_d0) 



    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == "__main__":

    model = ducknet(input_shape=(256,256,3), start_filters=17)
    model.summary()
