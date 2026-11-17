"""Compare une experience k-fold a une reference, fold par fold.

    python src/core/compare_kfold.py models/aslc_kfold_baseline.json \\
                                     models/aslc_kfold_layernorm90.json

La reference est notee sur la MEME fenetre que l experience : les 30
dernieres epochs du budget de l experience, lues dans les courbes sauvegardees
de la reference. Un run de 150 epochs contient exactement un run de 90 (pas de
taux d apprentissage qui depende du total, calcul reproductible au bit pres),
donc la baseline n a pas a etre relancee pour chaque budget.

Le test est APPARIE : chaque fold a ses propres signeurs de test, et l ecart
entre folds (2 points sur la baseline) est plus grand que la plupart des
effets cherches. Comparer les moyennes noierait l effet dans cet ecart ;
comparer fold a fold l annule.
"""

import argparse
import json
import math

import numpy as np

WINDOW = 30
# Au-dela de ce gain entre le debut et la fin de la fenetre notee, une
# courbe n a pas fini de monter : la noter sur cette fenetre la penaliserait.
# Le bruit d une epoch a l autre est de ~0.9 point ; moyenne sur 10 epochs et
# 5 folds, il tombe sous 0.2 point.
STILL_RISING = 0.005


def trend(data, budget):
    """Gain moyen (sur les folds) des 10 dernieres epochs de la fenetre sur
    ses 10 premieres."""
    curves = [np.asarray(c, dtype=float) for c in data["curves"]]
    start = budget - WINDOW
    return float(np.mean([c[budget - 10:budget].mean() - c[start:start + 10].mean()
                          for c in curves]))


def plateau(data, budget):
    """Moyenne des epochs budget-30 a budget, par fold, depuis les courbes."""
    curves = [np.asarray(c, dtype=float) for c in data["curves"]]
    short = [len(c) for c in curves if len(c) < budget]
    if short:
        raise SystemExit(f"courbes de {short[0]} epochs, budget demande {budget}")
    return np.array([c[budget - WINDOW:budget].mean() for c in curves])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("experiment")
    args = parser.parse_args()

    with open(args.reference, encoding="utf-8") as handle:
        reference = json.load(handle)
    with open(args.experiment, encoding="utf-8") as handle:
        experiment = json.load(handle)

    # Memes folds, memes signeurs : sinon l appariement ne veut rien dire.
    for key in ("folds", "vocab", "features_dir", "selection"):
        if reference.get(key) != experiment.get(key):
            raise SystemExit(f"{key} differe : {reference.get(key)!r} contre "
                             f"{experiment.get(key)!r} ; les folds ne se correspondent pas")

    budget = experiment["epochs"]
    ref = plateau(reference, budget)
    exp = plateau(experiment, budget)
    diff = exp - ref
    n = len(diff)
    sd = diff.std(ddof=1) if n > 1 else 0.0
    if sd > 0:
        t = diff.mean() / (sd / math.sqrt(n))
    else:
        # Ecarts tous egaux : nuls, rien a tester ; sinon un effet sans bruit.
        t = 0.0 if diff.mean() == 0 else math.copysign(float("inf"), diff.mean())
    try:
        from scipy import stats
        p = f"{2 * stats.t.sf(abs(t), n - 1):.3f}"
    except ImportError:
        p = "(scipy absent)"

    print(f"plateau sur les epochs {budget - WINDOW + 1}-{budget}\n")
    print("fold          " + "  ".join(f"{i:>7d}" for i in range(1, n + 1)) + "    moyenne")
    print("reference     " + "  ".join(f"{x:7.4f}" for x in ref) + f"    {ref.mean():.4f}")
    print("experience    " + "  ".join(f"{x:7.4f}" for x in exp) + f"    {exp.mean():.4f}")
    print("ecart         " + "  ".join(f"{x:+7.4f}" for x in diff) + f"    {diff.mean():+.4f}")
    print(f"\n{(diff > 0).sum()}/{n} folds en hausse ; t = {t:.2f}, p = {p} (test apparie, "
          f"{n - 1} ddl)")

    print(f"\nmontee dans la fenetre (10 dernieres epochs - 10 premieres) : "
          f"reference {trend(reference, budget):+.4f}, "
          f"experience {trend(experiment, budget):+.4f}")
    for name, data in (("reference", reference), ("experience", experiment)):
        if trend(data, budget) > STILL_RISING:
            print(f"ATTENTION : la {name} monte encore en fin de budget ; plus "
                  f"d epochs la departageraient plus justement.")


if __name__ == "__main__":
    main()
