"""Architecture du CNN pour la classification des signes."""

from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(64, 64, 3), num_classes=28):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Bloc 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Bloc 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Bloc 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Classification
        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            num_classes,
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model