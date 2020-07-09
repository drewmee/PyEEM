class CDAE():
    """[summary]
    """
    def __init__(self):
        return

def cdae():
    from keras.layers import Reshape, Conv2DTranspose, Cropping2D, MaxPooling2D, UpSampling2D
    # Network parameters
    input_shape = (142, 139, 1)

    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Cropping2D(cropping=((1, 1), (1, 0)))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=True, validation_split=0.3) #validation_data=(x_test_noisy, x_test))
