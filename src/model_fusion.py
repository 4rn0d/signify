"""Modele a deux branches : image (CNN) + landmarks MediaPipe.

Pourquoi fusionner plutot que choisir :

- Le CNN lit les pixels. Il est precis sur les differences fines de
  position du pouce (M / N / S / T), mais il surapprend l'apparence :
  eclairage, arriere-plan, camera. Mesure sur ce dataset, il passe de
  0.9938 (decoupage aleatoire, fuite) a 0.8277 (decoupage temporel).

- Les landmarks sont de la geometrie pure. Insensibles a l'eclairage et
  au fond, mais la profondeur z issue d'une seule camera est bruitee,
  donc les signes qui ne different que par un contact de doigts sont
  plus ambigus.

Les deux se trompent sur des choses differentes : les concatener donne
au classifieur les deux sources a la fois.

Cout a l'inference : quasi nul. HandSegmenter fait deja tourner MediaPipe
pour calculer la boite englobante — les landmarks sont deja calcules et
simplement jetes aujourd'hui.
"""

from tensorflow.keras import layers, models

from model_cnn import build_cnn_backbone
from model_landmark import build_landmark_branch


def build_fusion_model(
    image_shape=(64, 64, 3),
    landmark_dim=63,
    num_classes=28,
):
    image_input = layers.Input(shape=image_shape, name="image")
    landmark_input = layers.Input(shape=(landmark_dim,), name="landmarks")

    # Branche image : meme pile convolutive que le CNN seul.
    image_features = build_cnn_backbone(image_input)
    image_features = layers.Dense(128, activation="relu")(image_features)
    image_features = layers.Dropout(0.5)(image_features)

    # Branche landmarks : meme pile dense que le modele landmark seul.
    landmark_features = build_landmark_branch(landmark_input)

    # Fusion tardive : chaque branche resume son entree, puis on concatene.
    merged = layers.Concatenate()([image_features, landmark_features])

    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(
        inputs={"image": image_input, "landmarks": landmark_input},
        outputs=outputs,
        name="sign_fusion",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
