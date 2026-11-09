#!/usr/bin/env bash
# Lance un script Python du projet dans WSL, sur le GPU.
#
# TensorFlow n a plus de support GPU sous Windows natif depuis 2.11 ; sous
# WSL2, si. Seul l ENTRAINEMENT passe ici : l extraction (onnxruntime +
# TensorRT) et la demo restent sous Windows, et les modeles ecrits dans
# models/ y sont relus tels quels (memes versions de TensorFlow et de Keras
# des deux cotes : 2.21.0 et 3.15.1).
#
# Depuis Windows, a la racine du depot :
#   wsl -d Ubuntu-24.04 -- bash scripts/wsl_gpu.sh src/core/train_kfold.py --cell stgcn ...
#
# Environnement : ~/venvs/signify-gpu dans WSL, cree par
#   python3 -m venv ~/venvs/signify-gpu
#   ~/venvs/signify-gpu/bin/pip install 'tensorflow[and-cuda]==2.21.0' 'keras==3.15.1' 'numpy==2.4.6'
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venvs/signify-gpu/bin/activate
# TensorFlow ne trouve pas seul les bibliotheques CUDA installees par pip
# (nvidia-*-cu12) : il previent par un simple avertissement ("Cannot dlopen
# some GPU libraries"), que le niveau de log ci-dessous masque, et retombe
# sur le CPU sans erreur. On les met donc sur le chemin du chargeur.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
for lib in "$SITE"/nvidia/*/lib; do
  LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
# Les messages d initialisation de TensorFlow noient la sortie utile.
export TF_CPP_MIN_LOG_LEVEL=2
exec python "$@"
