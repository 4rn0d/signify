"""Architecture du CNN pour la classification des signes."""

from tensorflow.keras import layers, models


def build_cnn_backbone(inputs):
    """Blocs convolutifs, partages avec le modele de fusion.

    Extrait ici pour que le CNN seul et la branche image du modele de
    fusion ne puissent pas diverger : une seule definition a maintenir.
    """
    # Bloc 1
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 2
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 3
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)

    return layers.Flatten()(x)


def build_cnn_model(input_shape=(64, 64, 3), num_classes=28):
    inputs = layers.Input(shape=input_shape)

    x = build_cnn_backbone(inputs)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs, name="sign_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
