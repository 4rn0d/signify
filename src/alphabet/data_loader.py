"""Chargement et preparation du dataset d'images pour le CNN.

IMPORTANT - decoupage train/val
-------------------------------
Le dataset ASL Alphabet est constitue de frames video sequentielles
({classe}1.jpg ... {classe}3000.jpg). Deux frames voisines sont quasi
identiques : mesure sur 200 paires, l'ecart moyen par pixel est de 4.7
entre frames consecutives contre 61.0 entre frames tirees au hasard.

Un decoupage ALEATOIRE place donc les voisines d'une image de validation
dans le train : le modele reconnait des images deja vues plutot que le
signe. Mesure : 0.9938 en decoupage aleatoire contre 0.8277 en decoupage
temporel, pour un modele et un entrainement identiques.

On decoupe donc par INDICE de frame : les premiers 80% de chaque classe
pour le train, les derniers 20% pour la validation.
"""

import os
import re

import tensorflow as tf

IMG_SIZE = 64
BATCH_SIZE = 32

# Buffer de melange, exprime en nombre de BATCHES (le dataset est deja batche).
SHUFFLE_BUFFER = 256

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

_INDEX_RE = re.compile(r"(\d+)\.[A-Za-z]+$")


def build_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        # value_range doit decrire l'echelle reelle des pixels a ce stade :
        # ils sont normalises en [0, 1] par decode_image(), pas en [0, 255].
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.15),
    ])


def frame_index(path):
    """Indice de frame extrait du nom de fichier ('A123.jpg' -> 123).

    Retourne -1 si le nom ne contient pas de numero : ces fichiers sont
    alors tries par nom, de maniere deterministe.
    """
    match = _INDEX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def list_class_files(data_dir):
    """{classe: [chemins tries par indice de frame]}."""
    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    per_class = {}

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)

        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        # Tri par indice, puis par nom pour les fichiers sans numero.
        per_class[cls] = sorted(
            files,
            key=lambda p: (frame_index(p), os.path.basename(p)),
        )

    return per_class


def grouped_split(data_dir, val_split=0.2):
    """Decoupage temporel : debut de chaque classe en train, fin en val."""
    per_class = list_class_files(data_dir)
    class_names = sorted(per_class)

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    for label, cls in enumerate(class_names):
        files = per_class[cls]
        cut = int(len(files) * (1.0 - val_split))

        train_files += files[:cut]
        train_labels += [label] * cut

        val_files += files[cut:]
        val_labels += [label] * (len(files) - cut)

    return (
        (train_files, train_labels),
        (val_files, val_labels),
        class_names,
    )


def _decode(img_size):
    def fn(path, label):
        img = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False,
        )
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32) / 255.0, label
    return fn


def load_image_dataset(
    data_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.2,
    seed=123
):
    (train_files, train_labels), (val_files, val_labels), class_names = (
        grouped_split(data_dir, val_split=val_split)
    )

    print(f"Classes détectées ({len(class_names)}):")
    print(class_names)
    print(
        f"Découpage temporel : {len(train_files)} train / "
        f"{len(val_files)} val "
        f"(les images de validation sont les dernières frames de chaque classe)"
    )

    # Sécurité : empêcher un entraînement absurde
    if len(class_names) < 2:
        raise ValueError(
            "Une seule classe a été détectée. "
            "Vérifie DATA_DIR et la structure des dossiers."
        )

    if not train_files or not val_files:
        raise ValueError(
            "Découpage vide. Vérifie que chaque classe contient "
            "assez d'images."
        )

    decode = _decode(img_size)
    augmentation = build_augmentation_pipeline()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        .shuffle(
            len(train_files),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names
