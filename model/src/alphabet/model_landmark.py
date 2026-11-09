"""Reseau dense pour la classification a partir des landmarks MediaPipe."""

from tensorflow.keras import layers, models


def build_landmark_branch(inputs):
    """Branche landmarks, partagee avec le modele de fusion."""
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    return layers.Dense(64, activation="relu")(x)


def build_landmark_model(input_dim=63, num_classes=28):
    """input_dim = 21 landmarks x 3 coordonnees (x, y, z).

    Les coordonnees doivent etre normalisees au prealable
    (landmark_extractor.normalize_landmarks) : origine au poignet et
    echelle unitaire, sinon le modele apprend la position de la main
    dans le cadre plutot que la forme du signe.
    """
    inputs = layers.Input(shape=(input_dim,))

    x = build_landmark_branch(inputs)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="sign_landmark")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
