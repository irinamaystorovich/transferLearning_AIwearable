import tensorflow as tf


def create_lstm_model( num_classes=20,
                       dropout=0.8, units=128,
                       data_length=19,
                       optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 )
                    ):
    """
    Create a simple bidirectional LSTM model for classification. Creates a
    bidirectional LSTM layer, dropout, and two dense layers.

    :param num_classes: Number of classes.

    :param dropout: Dropout rate

    :param units: Number of units in LSTM layer.

    :param data_length: Length of time data for LSTM layer input.

    :param optimizer: Optimizer to use for model compilation.

    :return: The compiled model with LSTM, dropout, and two dense layers.
    """

    # Create the model object.
    model = tf.keras.models.Sequential()

    # Add an LSTM layer.
    # Input size is (19,3):
    #   19 time samples from data.
    #   3 dimensions (x, y, z accelerometer data).
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM( units=units, input_shape=[data_length, 3] )
        )
    )

    # Add dropout layer to reduce overfitting.
    model.add( tf.keras.layers.Dropout( rate=dropout ) )

    # Add final dense layers.
    model.add( tf.keras.layers.Dense( units=units, activation='relu' ) )
    model.add( tf.keras.layers.Dense( units=num_classes,
                                      activation='softmax' ) )

    model.compile( loss='sparse_categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'] )

    return model
