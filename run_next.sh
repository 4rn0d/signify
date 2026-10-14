#!/usr/bin/env bash
# Prochaine etape : la meilleure configuration nue (0.6462) avec toute la
# pile d entrainement, qui valait +8 points partout ailleurs.
#
#   reference nue        0.6462  (projection 48 + forme complete)
#   meilleure avec pile  0.6675  (sans projection ni forme complete)
#   attendu ici          ~0.70, non mesure
#
# Environ 2 h : 3 graines x (60 epochs de pre-entrainement + 150) x 5 folds.
set -e
cd "$(dirname "$0")"
python src/wlasl/train_kfold.py \
  --folds 5 --epochs 150 --vocab 50 --variants dominant \
  --features-dir data/wlasl/features_rtm_all \
  --rich --confidence --handshape-full \
  --gate 0.7 --gate-inside 0.5 --gate-conflict \
  --project 48 \
  --seeds 3 --pretrain-vocab 100 --label-smoothing 0.1 --tta \
  --vocab-from data/wlasl/features --clips-from data/wlasl/features \
  --tag _proj_everything
