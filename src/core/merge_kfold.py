"""Recolle les resultats d un run k-fold interrompu puis reprise avec --only-fold.

    python src/core/merge_kfold.py models/wlasl/wlasl_kfold_best.partial.json \
                                   models/wlasl/wlasl_kfold_fold5.json \
                                   --out models/wlasl/wlasl_kfold_best.json

Deux runs de ce projet sont morts en cours, le second tue par un redemarrage
Windows au cinquieme fold sur cinq. --only-fold permet de ne refaire que les
folds manquants ; ce script en rassemble les morceaux.

Le vrai travail n est pas la concatenation, c est le refus de concatener :
fusionner des folds venus de configurations differentes produirait une
moyenne qui ne decrit aucun modele, et rien dans le fichier resultant ne le
signalerait. Toute cle de configuration qui differe arrete donc le script.
"""

import argparse
import json
import os

import numpy as np

# Ce qui definit la tache et le modele. Deux fichiers qui divergent sur l une
# de ces cles ne mesurent pas la meme chose et ne se fusionnent pas.
CONFIG_KEYS = (
    "folds", "epochs", "units", "project", "glosses", "cell", "variants",
    "velocity", "no_z", "depth_proxy", "rich", "label_smoothing", "tta",
    "seeds", "pretrain_vocab", "pretrain_variants", "gate_pretrain",
    "pretrain_epochs", "shared_pretrain", "vocab", "confidence", "symmetry",
    "handshape", "handshape_full", "gate", "gate_inside", "gate_conflict",
    "gate_soft", "interpolate", "smooth", "tip_confidence", "max_per_class",
    "max_train_signers", "hand_dropout",
    # Un fold WLASL et des folds ASL Citizen partagent toute la configuration
    # ci-dessus ; seul le dossier de features les distingue.
    "features_dir", "selection",
    # Le GPU et le CPU ne calculent pas les memes nombres : un fold GPU recolle
    # a des folds CPU donnerait une moyenne de deux experiences differentes.
    "device",
    # Comment le GRU rend sa reponse, et ce qui le stabilise : un fold a etat
    # final et un fold a attention ne sont pas le meme modele.
    "readout", "layer_norm", "clipnorm",
)


def fold_numbers(data, path):
    """Les folds couverts par un fichier, 1-based.

    folds_run est ecrit depuis l ajout de --only-fold. Les fichiers plus
    anciens n en ont pas : on suppose alors qu ils commencent au fold 1, ce
    qui est vrai d un run complet comme d un partiel, les deux s arretant
    simplement plus ou moins loin.
    """
    if "folds_run" in data:
        return list(data["folds_run"])
    count = len(data.get("per_fold", []))
    print(f"  {os.path.basename(path)} : pas de folds_run, "
          f"on suppose les folds 1..{count}")
    return list(range(1, count + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="resultats a fusionner")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="ecrire meme s il manque des folds")
    parser.add_argument("--replace-fold", type=int, action="append", default=[],
                        metavar="N",
                        help="le fold N present dans plusieurs fichiers est pris "
                             "dans le DERNIER, et la substitution est ecrite dans "
                             "le resultat. Sert a refaire un fold rate : le fold 5 "
                             "de la baseline ASL Citizen, effondre a la graine "
                             "500, refait a la graine 501.")
    args = parser.parse_args()

    loaded = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            loaded.append((path, json.load(handle)))

    # --- la configuration doit etre la meme partout ---
    reference_path, reference = loaded[0]
    for path, data in loaded[1:]:
        for key in CONFIG_KEYS:
            if key in reference and key in data and reference[key] != data[key]:
                raise SystemExit(
                    f"configurations differentes, fusion refusee :\n"
                    f"  {key} = {reference[key]!r} dans {reference_path}\n"
                    f"  {key} = {data[key]!r} dans {path}")

    # --- un fold ne peut venir que d un seul fichier ---
    by_fold = {}
    replaced = {}
    for path, data in loaded:
        numbers = fold_numbers(data, path)
        for position, fold in enumerate(numbers):
            if fold in by_fold:
                if fold not in args.replace_fold:
                    raise SystemExit(
                        f"le fold {fold} est present dans deux fichiers : "
                        f"{by_fold[fold][0]} et {path} (--replace-fold {fold} "
                        f"pour prendre le second)")
                replaced[str(fold)] = {
                    "dropped_from": os.path.basename(by_fold[fold][0]),
                    "taken_from": os.path.basename(path),
                    "seed": data.get("seed")}
            by_fold[fold] = (path, data, position)
    unused = sorted(set(args.replace_fold) - {int(k) for k in replaced})
    if unused:
        raise SystemExit(f"--replace-fold {unused} : ces folds n apparaissent "
                         f"que dans un fichier, rien a remplacer")

    total = reference.get("folds", max(by_fold))
    missing = [f for f in range(1, total + 1) if f not in by_fold]
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"folds manquants : {missing}. Relance-les avec "
            f"--only-fold {','.join(map(str, missing))}, ou passe "
            f"--allow-incomplete.")

    order = sorted(by_fold)
    merged = dict(reference)

    # out_of_fold porte les predictions de chaque fold (matrice de confusion) ;
    # pretrain_curves la courbe de pre-entrainement de chaque fold.
    for field in ("per_fold", "per_fold_plateau", "curves", "by_seed_count",
                  "out_of_fold", "pretrain_curves"):
        values = []
        for fold in order:
            _, data, position = by_fold[fold]
            if field in data and position < len(data[field]):
                values.append(data[field][position])
        if values:
            merged[field] = values

    plateau = np.array(merged["per_fold_plateau"], dtype=float)
    merged["plateau_mean"] = float(plateau.mean())
    merged["plateau_std"] = float(plateau.std())
    scores = np.array(merged["per_fold"], dtype=float)
    merged["mean"] = float(scores.mean())
    merged["std"] = float(scores.std())
    merged["folds_run"] = order
    merged["merged_from"] = [os.path.basename(p) for p, _ in loaded]
    if replaced:
        merged["replaced_folds"] = replaced
    merged.pop("folds_done", None)
    merged.pop("only_fold", None)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)

    print(f"  folds {order} fusionnes depuis {len(loaded)} fichiers")
    print(f"  plateau {plateau.mean():.4f}  ecart-type {plateau.std():.4f}")
    print(f"  par fold " + " ".join(f"{x:.4f}" for x in plateau))
    if missing:
        print(f"  INCOMPLET : folds {missing} absents")
    print(f"  ecrit dans {args.out}")


if __name__ == "__main__":
    main()
