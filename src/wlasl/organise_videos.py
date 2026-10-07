"""Range les videos par glose au lieu d'un seul dossier d'identifiants.

Avant : data/wlasl/videos/69241.mp4        (illisible)
Apres : data/wlasl/videos/drink/drink_s118_69241.mp4

Le nom porte les trois informations utiles pour inspecter le dataset :
la glose, le signeur (les decoupages sont faits PAR signeur, donc c'est
l'axe qui compte pour comprendre une erreur) et l'identifiant d'origine,
qui reste la cle du manifeste et des features.

Le manifeste est mis a jour en meme temps : c'est lui la source de verite
pour l'extraction, pas le chemin sur disque. Le script est idempotent, on
peut le relancer sans risque.

Les features (.npz) restent a plat : elles portent deja leur glose a
l'interieur, et le chargeur les parcourt par glob.
"""

import argparse
import collections
import json
import os
import re
import shutil


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")
VIDEO_DIR = os.path.join(WLASL_DIR, "videos")

SAFE = re.compile(r"[^A-Za-z0-9_-]")


def target_name(record):
    gloss = SAFE.sub("", record["gloss"].replace(" ", "_"))
    return gloss, f"{gloss}_s{record['signer_id']}_{record['video_id']}.mp4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flatten", action="store_true",
                        help="operation inverse : tout remettre a plat")
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)

    moved = 0
    missing = 0
    already = 0
    per_gloss = collections.Counter()

    for record in manifest:
        if record["status"] not in ("ok", "cached") or not record["path"]:
            continue

        current = os.path.join(PROJECT_ROOT, record["path"])

        if args.flatten:
            destination = os.path.join(VIDEO_DIR, f"{record['video_id']}.mp4")
        else:
            gloss, name = target_name(record)
            destination = os.path.join(VIDEO_DIR, gloss, name)
            per_gloss[gloss] += 1

        if os.path.abspath(current) == os.path.abspath(destination):
            already += 1
            record["path"] = os.path.relpath(destination, PROJECT_ROOT)
            continue

        if not os.path.exists(current):
            # Peut-etre deja deplace lors d'un passage precedent.
            if os.path.exists(destination):
                already += 1
                record["path"] = os.path.relpath(destination, PROJECT_ROOT)
            else:
                missing += 1
            continue

        if not args.dry_run:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(current, destination)

        record["path"] = os.path.relpath(destination, PROJECT_ROOT)
        moved += 1

    if not args.dry_run:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=1)

        # Supprime les dossiers de gloses devenus vides apres un --flatten.
        for entry in os.listdir(VIDEO_DIR):
            path = os.path.join(VIDEO_DIR, entry)
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)

    verb = "a deplacer" if args.dry_run else "deplaces"
    print(f"  {verb:<14}{moved:>6}")
    print(f"  {'deja en place':<14}{already:>6}")
    print(f"  {'introuvables':<14}{missing:>6}")

    if per_gloss and not args.flatten:
        print(f"\n  {len(per_gloss)} dossiers de gloses")
        top = per_gloss.most_common()
        print(f"  le plus fourni : {top[0][0]} ({top[0][1]})")
        print(f"  le moins       : {top[-1][0]} ({top[-1][1]})")

    if not args.dry_run:
        print(f"\n  manifeste mis a jour : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
