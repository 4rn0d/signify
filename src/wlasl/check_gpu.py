"""Le GPU calcule-t-il vraiment, et de facon reproductible ?

A lancer dans WSL :  bash scripts/wsl_gpu.sh src/wlasl/check_gpu.py

1. TensorFlow ne se contente pas de LISTER le GPU : un calcul y est place et
   son resultat est juste. Le 30 octobre, onnxruntime listait CUDA et
   tournait sur CPU ; une liste ne prouve rien.
2. Avec --deterministic, deux entrainements identiques donnent des poids
   identiques au bit pres, pour chaque architecture. C est ce qui garde
   --only-fold exact sur GPU.
3. Le temps de chargement des features a travers le pont Windows / Linux
   (/mnt/c), pour decider s il faut les copier cote Linux.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tensorflow as tf  # noqa: E402

from train_kfold import build_model, compute_device, load_sequences  # noqa: E402

device = compute_device(deterministic=True)
gpus = tf.config.list_physical_devices("GPU")
print(f"1. appareil : {device}   GPU listes : {[g.name for g in gpus]}")
if not gpus:
    raise SystemExit("   aucun GPU : rien a verifier de plus")

with tf.device("/GPU:0"):
    a = tf.random.uniform((2048, 2048), seed=0)
    product = tf.linalg.matmul(a, a)
print(f"   calcul place sur : {product.device}")
reference = np.matmul(a.numpy().astype(np.float64), a.numpy().astype(np.float64))
error = float(np.abs(product.numpy() - reference).max() / np.abs(reference).max())
print(f"   ecart relatif avec numpy : {error:.1e}")

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 80, 155)).astype(np.float32)
y = tf.keras.utils.to_categorical(rng.integers(0, 100, 256), 100)
print("\n2. deux entrainements identiques, --deterministic :")
for cell in ("gru", "tcn", "stgcn"):
    runs = []
    for _ in range(2):
        tf.keras.utils.set_random_seed(0)
        model = build_model(100, 96, 155, cell, 0.1, 48, raw_dim=106)
        model.fit(x, y, batch_size=32, epochs=2, verbose=0, shuffle=False)
        runs.append(model.get_weights())
    same = all(np.array_equal(a, b) for a, b in zip(*runs))
    print(f"   {cell:6s} poids identiques au bit : {same}")

features = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "asl_citizen", "features")
start = time.perf_counter()
records = load_sequences("all", features, with_confidence=True, with_meta=True)
seconds = time.perf_counter() - start
print(f"\n3. chargement de {len(records)} clips depuis {features[:12]}... : {seconds:.0f} s "
      f"({seconds / max(len(records), 1) * 1000:.1f} ms par clip)")
