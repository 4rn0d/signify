"""Audit du metadata WLASL avant tout telechargement.

Trois questions, dans l'ordre d'importance :

1. Les decoupages officiels separent-ils les SIGNEURS ? Si un meme signeur
   apparait en train et en test, le modele peut reconnaitre la personne
   plutot que le signe. C'est exactement l'erreur qui donnait 0.9938 sur
   l'alphabet (voir docs/FINDINGS.md section 4) — en pire, parce qu'un
   signeur est plus facile a memoriser qu'une lettre.

2. Combien de clips par glose reellement ? Le papier annonce des moyennes,
   mais c'est le minimum par classe qui decide si l'entrainement tient.

3. D'ou viennent les videos ? Les liens morts sont le probleme pratique
   numero un de WLASL, et ils ne tombent pas uniformement selon l'hote.

Aucun telechargement ici : uniquement le JSON.
"""

import argparse
import collections
import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "wlasl", "WLASL_v0.3.json")

# Les sous-ensembles WLASL (100 / 300 / 1000 / 2000) sont les N gloses ayant
# le plus d'instances.
SUBSET_SIZES = (100, 300, 1000, 2000)


def load(path=METADATA_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def subset(data, size):
    """Les `size` gloses les mieux fournies, comme dans le papier."""
    ranked = sorted(data, key=lambda g: len(g["instances"]), reverse=True)
    return ranked[:size]


def describe_sizes(data):
    print("=" * 70)
    print("1. TAILLE DES SOUS-ENSEMBLES")
    print("=" * 70)
    print(f"  {'subset':<10}{'gloses':>8}{'clips':>9}{'min/glose':>11}"
          f"{'median':>9}{'max':>7}")

    for size in SUBSET_SIZES:
        glosses = subset(data, size)
        counts = sorted(len(g["instances"]) for g in glosses)
        total = sum(counts)
        median = counts[len(counts) // 2]
        print(f"  WLASL{size:<5}{len(glosses):>8}{total:>9}"
              f"{counts[0]:>11}{median:>9}{counts[-1]:>7}")


def describe_splits(data, size=100):
    print()
    print("=" * 70)
    print(f"2. DECOUPAGES OFFICIELS (WLASL{size})")
    print("=" * 70)

    glosses = subset(data, size)
    per_split = collections.Counter()
    signers_by_split = collections.defaultdict(set)

    for gloss in glosses:
        for inst in gloss["instances"]:
            per_split[inst["split"]] += 1
            signers_by_split[inst["split"]].add(inst["signer_id"])

    total = sum(per_split.values())
    for name in ("train", "val", "test"):
        count = per_split.get(name, 0)
        print(f"  {name:<8}{count:>7} clips ({count / total:>5.1%})"
              f"   {len(signers_by_split[name]):>3} signeurs distincts")

    return signers_by_split


def check_signer_leakage(signers_by_split):
    print()
    print("=" * 70)
    print("3. FUITE DE SIGNEURS  <-- la question qui decide de tout")
    print("=" * 70)

    train = signers_by_split["train"]
    val = signers_by_split["val"]
    test = signers_by_split["test"]

    overlap_val = train & val
    overlap_test = train & test

    print(f"  signeurs en train           : {len(train)}")
    print(f"  signeurs en val             : {len(val)}")
    print(f"  signeurs en test            : {len(test)}")
    print()
    print(f"  presents en train ET val    : {len(overlap_val)}"
          f"  ({len(overlap_val) / max(len(val), 1):.0%} des signeurs de val)")
    print(f"  presents en train ET test   : {len(overlap_test)}"
          f"  ({len(overlap_test) / max(len(test), 1):.0%} des signeurs de test)")
    print()

    if overlap_test:
        print("  => Les decoupages officiels ne sont PAS independants du")
        print("     signeur. Un score obtenu dessus mesure en partie la")
        print("     reconnaissance de personnes, pas de signes.")
        print("     Prevoir un decoupage par signeur pour toute mesure honnete.")
    else:
        print("  => Decoupages independants du signeur : utilisables tels quels.")


def describe_sources(data, size=100):
    print()
    print("=" * 70)
    print(f"4. HEBERGEURS (WLASL{size}) — risque de liens morts")
    print("=" * 70)

    sources = collections.Counter()
    for gloss in subset(data, size):
        for inst in gloss["instances"]:
            sources[inst["source"]] += 1

    total = sum(sources.values())
    for name, count in sources.most_common():
        print(f"  {name:<16}{count:>6} ({count / total:>5.1%})")


def describe_clips(data, size=100):
    print()
    print("=" * 70)
    print(f"5. CLIPS (WLASL{size})")
    print("=" * 70)

    fps = collections.Counter()
    trimmed = 0
    total = 0

    for gloss in subset(data, size):
        for inst in gloss["instances"]:
            fps[inst["fps"]] += 1
            total += 1
            if inst["frame_end"] != -1:
                trimmed += 1

    print("  fps :", ", ".join(f"{k} ({v})" for k, v in fps.most_common(5)))
    print(f"  clips avec un intervalle de frames explicite : {trimmed}/{total}")
    print("  (frame_end = -1 signifie : prendre la video entiere)")

    worst = sorted(subset(data, size), key=lambda g: len(g["instances"]))[:8]
    print()
    print("  gloses les moins fournies du sous-ensemble :")
    for gloss in worst:
        print(f"    {gloss['gloss']:<18}{len(gloss['instances']):>3} clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--metadata", default=METADATA_PATH)
    args = parser.parse_args()

    data = load(args.metadata)

    print(f"\nWLASL : {len(data)} gloses, "
          f"{sum(len(g['instances']) for g in data)} instances\n")

    describe_sizes(data)
    signers = describe_splits(data, args.subset)
    check_signer_leakage(signers)
    describe_sources(data, args.subset)
    describe_clips(data, args.subset)
    print()


if __name__ == "__main__":
    main()
