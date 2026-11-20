import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense
)


# --------------------------------------------------
# Build vibration LSTM model
# --------------------------------------------------

def build_vibration_lstm(
    time_steps=16,
    n_features=5,
    n_classes=3
):

    model = Sequential([

        # Input layer
        Input(shape=(time_steps, n_features)),

        # First LSTM layer
        LSTM(64, return_sequences=True),

        # Dropout to reduce overfitting
        Dropout(0.2),

        # Second LSTM layer
        LSTM(32),

        # Fully connected layer
        Dense(16, activation="relu"),

        # Three-class output
        Dense(n_classes, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# Test model architecture
# --------------------------------------------------

if __name__ == "__main__":

    model = build_vibration_lstm(time_steps=16)

    model.summary()