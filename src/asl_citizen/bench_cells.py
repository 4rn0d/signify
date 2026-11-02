"""GRU, TCN, ST-GCN : taille et cout d une epoch, sur les vrais clips cibles.

Avant d engager des heures de k-fold, deux chiffres par architecture :
  - le nombre de parametres, pour que la comparaison porte sur
    l architecture et pas sur la taille ;
  - le temps d une epoch d entrainement sur ~80% de la cible, ce qu un fold
    entraine, puisque c est ce qui decide combien d experiences on peut
    s offrir par nuit.

Et deux verifications qui ne cassent qu au pire moment sinon :
  - un modele sauvegarde se RECHARGE (la demo en depend, et le ST-GCN a des
    couches maison) et predit la meme chose ;
  - le transfert du pre-entrainement, couche par couche sauf la derniere,
    tombe sur des formes identiques.

Les temps sont mesures pendant que l extraction de l echafaudage tourne :
ils sont gonfles, mais de la meme facon pour les trois.
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "wlasl"))

import tensorflow as tf  # noqa: E402

from train_kfold import (Layout, add_handshape, add_rich_features,  # noqa: E402
                         augment_batch, build_model, gate_hands, load_sequences)

FEATURES = os.path.join(ROOT, "data", "asl_citizen", "features")
rng = np.random.default_rng(0)

records = [r for r in load_sequences("all", FEATURES, with_confidence=True, with_meta=True)
           if r["rank"] is not None and r["rank"] <= 100]
glosses = sorted({r["gloss"] for r in records})
index = {g: i for i, g in enumerate(glosses)}
features = np.stack([r["features"] for r in records])
labels = np.array([index[r["gloss"]] for r in records])

layout = Layout.from_dim(features.shape[2] - 2)
layout.tail = 2
meta = lambda key: np.stack([r["meta"][key] for r in records])  # noqa: E731
features = gate_hands(features, layout, meta("hand_scores"), meta("hand_inside"),
                      meta("hand_conflict"), threshold=0.7, inside_threshold=0.5,
                      use_conflict=True)
raw_dim = features.shape[2]


def to_model(batch):
    return np.concatenate([batch, add_rich_features(batch, layout)[:, :, -23:],
                           add_handshape(batch, layout, full=True)[:, :, -26:]], axis=-1)


train = rng.permutation(len(features))[: int(0.8 * len(features))]
x = to_model(augment_batch(features[train], rng, layout))
y = tf.keras.utils.to_categorical(labels[train], len(glosses))
feature_dim = x.shape[2]
print(f"{len(records)} clips, {len(glosses)} signes ; une epoch = {len(train)} clips, "
      f"entree {x.shape[1:]} (dont {raw_dim} colonnes brutes)")
print()
print(f"  {'cellule':7s} {'parametres':>11s} {'s / epoch':>10s} {'rechargement':>13s} "
      f"{'transfert':>10s}")

for cell in ("gru", "tcn", "stgcn"):
    tf.keras.utils.set_random_seed(0)
    model = build_model(len(glosses), 96, feature_dim, cell, 0.1, 48, raw_dim=raw_dim)
    params = model.count_params()

    model.fit(x[:256], y[:256], batch_size=32, epochs=1, verbose=0)   # trace, non compte
    start = time.perf_counter()
    model.fit(x, y, batch_size=32, epochs=1, verbose=0)
    seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, f"{cell}.keras")
        model.save(path)
        reloaded = tf.keras.models.load_model(path)
        same = np.allclose(model.predict(x[:8], verbose=0),
                           reloaded.predict(x[:8], verbose=0), atol=1e-6)

    head = build_model(400, 96, feature_dim, cell, 0.1, 48, raw_dim=raw_dim)
    fits = all(
        [w.shape for w in target.get_weights()] == [w.shape for w in source.get_weights()]
        for target, source in zip(model.layers[:-1], head.layers[:-1]))

    print(f"  {cell:7s} {params:11,d} {seconds:10.1f} {'identique' if same else 'DIFFERENT':>13s} "
          f"{'ok' if fits else 'CASSE':>10s}")
