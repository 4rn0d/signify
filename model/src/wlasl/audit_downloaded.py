"""Audit de ce qui a REELLEMENT ete telecharge, par glose et par signeur.

Le metadata annonce 2038 clips pour WLASL100 ; les liens morts en retirent
une grande partie. Ce qui compte pour decider de la suite, ce n'est pas le
total mais la distribution :

- combien de gloses tombent sous le seuil utilisable ?
- un decoupage independant du signeur reste-t-il possible ?

Les decoupages officiels de WLASL partagent 93% de leurs signeurs entre
train et test (voir audit_metadata.py), ils ne sont donc pas utilisables
tels quels pour une mesure honnete.
"""

import argparse
import collections
import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "wlasl", "download_manifest.json")

# En dessous de ce nombre de clips, une glose n'a pas de quoi apprendre ET
# etre evaluee de maniere credible.
MIN_USABLE = 6


def load(path=MANIFEST_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def available(manifest):
    return [r for r in manifest if r["status"] in ("ok", "cached")]


def report_overall(manifest, have):
    print("=" * 68)
    print("1. RECUPERATION")
    print("=" * 68)

    counts = collections.Counter(r["status"] for r in manifest)
    total = len(manifest)

    for status in ("ok", "cached", "failed", "skipped_dead_host"):
        if counts[status]:
            print(f"  {status:<20}{counts[status]:>6} ({counts[status] / total:>5.1%})")

    print(f"  {'-' * 32}")
    print(f"  {'utilisables':<20}{len(have):>6} ({len(have) / total:>5.1%})")


def report_per_gloss(have):
    print()
    print("=" * 68)
    print("2. CLIPS PAR GLOSE")
    print("=" * 68)

    per_gloss = collections.Counter(r["gloss"] for r in have)
    counts = sorted(per_gloss.values())

    print(f"  gloses representees : {len(per_gloss)}/100")
    print(f"  min {counts[0]}   mediane {counts[len(counts) // 2]}   max {counts[-1]}")
    print()

    for threshold in (3, MIN_USABLE, 10):
        under = sum(1 for c in counts if c < threshold)
        print(f"  gloses avec moins de {threshold:>2} clips : {under:>3}")

    worst = sorted(per_gloss.items(), key=lambda kv: kv[1])[:10]
    print()
    print("  les moins fournies :")
    for gloss, count in worst:
        flag = "  <-- sous le seuil" if count < MIN_USABLE else ""
        print(f"    {gloss:<16}{count:>3}{flag}")

    return per_gloss


def report_signer_split(have, per_gloss):
    print()
    print("=" * 68)
    print("3. DECOUPAGE INDEPENDANT DU SIGNEUR")
    print("=" * 68)

    signers = collections.Counter(r["signer_id"] for r in have)
    print(f"  signeurs distincts : {len(signers)}")
    print(f"  clips par signeur  : mediane "
          f"{sorted(signers.values())[len(signers) // 2]}, max {max(signers.values())}")

    # On retient les signeurs les plus rares en premier : cela libere ~20%
    # des clips sans vider une classe de ses rares exemples.
    held = set()
    held_clips = 0
    target = len(have) * 0.2

    for signer, count in sorted(signers.items(), key=lambda kv: kv[1]):
        if held_clips >= target:
            break
        held.add(signer)
        held_clips += count

    per_gloss_split = collections.defaultdict(lambda: [0, 0])
    for record in have:
        per_gloss_split[record["gloss"]][1 if record["signer_id"] in held else 0] += 1

    no_test = [g for g, (tr, te) in per_gloss_split.items() if te == 0]
    no_train = [g for g, (tr, te) in per_gloss_split.items() if tr == 0]
    thin = [g for g, (tr, te) in per_gloss_split.items() if 0 < te < 3]

    print()
    print(f"  signeurs retenus pour le test : {len(held)}/{len(signers)}")
    print(f"  clips de test                 : {held_clips} ({held_clips / len(have):.1%})")
    print()
    print(f"  gloses sans clip de test  : {len(no_test):>3}"
          f"{'   <-- bloquant' if no_test else ''}")
    print(f"  gloses sans clip de train : {len(no_train):>3}"
          f"{'   <-- bloquant' if no_train else ''}")
    print(f"  gloses avec 1-2 clips test: {len(thin):>3}   "
          "(precision par classe non significative dessus)")

    if no_test:
        print(f"    exemples : {', '.join(sorted(no_test)[:6])}")


def report_clips(have):
    print()
    print("=" * 68)
    print("4. CLIPS")
    print("=" * 68)

    trimmed = sum(1 for r in have if r["frame_end"] != -1)
    print(f"  avec intervalle de frames explicite : {trimmed}/{len(have)}")
    print("  (les autres : prendre la video entiere)")

    fps = collections.Counter(r["fps"] for r in have)
    print(f"  fps : {', '.join(f'{k} ({v})' for k, v in fps.most_common(4))}")

    sizes = []
    for record in have:
        path = os.path.join(PROJECT_ROOT, record["path"]) if record["path"] else None
        if path and os.path.exists(path):
            sizes.append(os.path.getsize(path))

    if sizes:
        sizes.sort()
        print(f"  taille : mediane {sizes[len(sizes) // 2] / 1024:.0f} Ko, "
              f"total {sum(sizes) / 1e6:.0f} Mo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    args = parser.parse_args()

    manifest = load(args.manifest)
    have = available(manifest)

    print()
    report_overall(manifest, have)
    per_gloss = report_per_gloss(have)
    report_signer_split(have, per_gloss)
    report_clips(have)
    print()


if __name__ == "__main__":
    main()
