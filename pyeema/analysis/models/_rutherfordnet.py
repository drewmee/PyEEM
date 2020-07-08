class RutherfordNet():
    """Summary
    """
    def __init__(self):
        return

    def create_model(self):
        """Create the model
        """
        return

    def train():
        return

    def create_training_data(self):
        return



def rutherfordnet(X, y):
    NAME = "rutherford-net-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

    model = Sequential()

    # Convolution layers
    # first layer
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=[142, 139, 1],
                     activation="elu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))

    # second layer
    model.add(Conv2D(10, (10, 10), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))

    # third layer
    model.add(Conv2D(10, (15, 15), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())  # converts 3D feature maps to 1D feature maps
    model.add(Dropout(0.2))

    # Dense Layers
    model.add(Dense(512, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='elu'))

    # Output layer
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='linear'))

    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=["accuracy"])

    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True,
                              show_layer_names=True, to_file='model.png')

    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3,
              shuffle=True, callbacks=[tensorboard])
    model.save('rutherford-net.h5')
    return model