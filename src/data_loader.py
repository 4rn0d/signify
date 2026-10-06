"""Chargement et preparation du dataset d'images pour le CNN."""

import tensorflow as tf

IMG_SIZE = 64
BATCH_SIZE = 32

# Buffer de melange, exprime en nombre de BATCHES (le dataset est deja batche).
SHUFFLE_BUFFER = 256


def build_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        # value_range doit decrire l'echelle reelle des pixels a ce stade :
        # ils sont deja normalises en [0, 1], pas en [0, 255].
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.15),
    ])


def load_image_dataset(
    data_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.2,
    seed=123
):
    common_kwargs = dict(
        validation_split=val_split,
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        # "int" des deux cotes : le modele utilise sparse_categorical_crossentropy.
        label_mode="int",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        subset="training",
        shuffle=True,
        **common_kwargs
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        subset="validation",
        shuffle=False,
        **common_kwargs
    )

    class_names = train_ds.class_names

    print(f"Classes détectées ({len(class_names)}):")
    print(class_names)

    # Sécurité : empêcher un entraînement absurde
    if len(class_names) < 2:
        raise ValueError(
            "Une seule classe a été détectée. "
            "Vérifie DATA_DIR et la structure des dossiers."
        )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = build_augmentation_pipeline()

    def to_uint8(x, y):
        # On met le cache en uint8 plutot qu'en float32 : ~4x moins de RAM
        # (0.9 Go au lieu de 3.5 Go pour ce dataset).
        return tf.saturate_cast(tf.round(x), tf.uint8), y

    def normalize(x, y):
        return normalization_layer(x), y

    def augment(x, y):
        return augmentation(x, training=True), y

    train_ds = (
        train_ds
        .map(to_uint8, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        # .cache() fige l'ordre produit en amont : sans ce shuffle, chaque epoch
        # rejouerait exactement la meme sequence de batches.
        .shuffle(SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True)
        .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(to_uint8, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names
