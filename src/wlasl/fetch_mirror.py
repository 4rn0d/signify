"""Complete le dataset depuis le miroir Hugging Face Voxel51/WLASL.

Pourquoi un miroir : les URL d'origine de WLASL pointent vers 19 sites, dont
5 sont morts (783 clips sur 2038) et YouTube bloque une partie du reste. Le
telechargement direct plafonne a 880 clips sur 2038 (43%).

Le miroir contient les fichiers video eux-memes, archives a une epoque ou
davantage de liens vivaient. Il n'est pas complet non plus (11 880 clips sur
21 083, soit 56%), mais les survivants ne sont PAS les memes : en cumulant
les deux sources on passe a 1321 clips sur 2038 (65%), et surtout de 50 a 73
signeurs — c'est la diversite de signeurs, pas le nombre de clips, qui
bloquait un decoupage honnete.

On ne telecharge que les clips manquants : inutile de rapatrier 11 880
fichiers pour en utiliser quelques centaines.

Licence : WLASL est publie sous C-UDA (usage academique et computationnel,
pas commercial).
"""

import argparse
import collections
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, list_repo_files


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
METADATA_PATH = os.path.join(WLASL_DIR, "WLASL_v0.3.json")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")
VIDEO_DIR = os.path.join(WLASL_DIR, "videos")

REPO_ID = "Voxel51/WLASL"
REPO_TYPE = "dataset"

# Le cache Hugging Face s'appuie sur des liens symboliques, que OneDrive ne
# supporte pas : chaque fichier serait donc stocke en double dans le projet.
# On le met hors du projet et on le supprime a la fin.
HF_CACHE = os.path.join(tempfile.gettempdir(), "wlasl_hf_cache")


def build_index():
    """{video_id sans zeros de tete: chemin dans le depot}."""
    index = {}
    for name in list_repo_files(REPO_ID, repo_type=REPO_TYPE):
        if name.endswith(".mp4"):
            stem = os.path.splitext(os.path.basename(name))[0]
            index[stem.lstrip("0")] = name
    return index


def load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def missing_records(manifest, index):
    """Clips absents en local mais presents dans le miroir."""
    out = []
    for record in manifest:
        if record["status"] in ("ok", "cached"):
            continue
        key = record["video_id"].lstrip("0")
        if key in index:
            out.append((record, index[key]))
    return out


def fetch_one(args):
    record, repo_path = args
    target = os.path.join(VIDEO_DIR, f"{record['video_id']}.mp4")

    if os.path.exists(target) and os.path.getsize(target) > 2048:
        return record, "cached", os.path.getsize(target)

    try:
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=repo_path,
            cache_dir=HF_CACHE,
        )

        size = os.path.getsize(cached)
        if size < 2048:
            raise ValueError(f"fichier trop petit ({size} octets)")

        # Copie sous le nom attendu par le reste du pipeline (video_id.mp4),
        # ecriture atomique pour qu'un fichier partiel ne passe pas pour bon.
        tmp = target + ".part"
        with open(cached, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, target)

        return record, "ok", size
    except Exception as error:
        return record, "failed", f"{type(error).__name__}: {str(error)[:110]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f"Lecture de la liste des fichiers de {REPO_ID}...", flush=True)
    index = build_index()
    print(f"  {len(index)} videos dans le miroir")

    manifest = load_manifest()
    todo = missing_records(manifest, index)
    if args.limit:
        todo = todo[: args.limit]

    already = sum(1 for r in manifest if r["status"] in ("ok", "cached"))
    print(f"  {already} clips deja en local")
    print(f"  {len(todo)} recuperables depuis le miroir\n")

    if not todo:
        print("Rien a faire.")
        return

    counts = collections.Counter()
    updated = {}
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, item) for item in todo]

        for index_done, future in enumerate(as_completed(futures), start=1):
            record, status, detail = future.result()
            counts[status] += 1
            if status in ("ok", "cached"):
                updated[record["video_id"]] = os.path.relpath(
                    os.path.join(VIDEO_DIR, f"{record['video_id']}.mp4"), PROJECT_ROOT
                )

            if index_done % 50 == 0 or index_done == len(todo):
                elapsed = time.time() - started
                rate = index_done / max(elapsed, 1e-6)
                print(
                    f"  {index_done}/{len(todo)}  ok {counts['ok']}  "
                    f"echec {counts['failed']}   "
                    f"~{(len(todo) - index_done) / max(rate, 1e-6) / 60:.0f} min restantes",
                    flush=True,
                )

    # Mise a jour du manifeste : les clips recuperes passent a "ok".
    for record in manifest:
        if record["video_id"] in updated:
            record["status"] = "ok"
            record["detail"] = "mirror:huggingface"
            record["path"] = updated[record["video_id"]]

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)

    shutil.rmtree(HF_CACHE, ignore_errors=True)

    total = sum(1 for r in manifest if r["status"] in ("ok", "cached"))
    print(f"\n{'=' * 60}")
    print(f"  recuperes depuis le miroir : {counts['ok']}")
    print(f"  echecs                     : {counts['failed']}")
    print(f"  total utilisable           : {total}/{len(manifest)} "
          f"({total / len(manifest):.0%})")
    print(f"\n  Relance audit_downloaded.py pour la distribution reelle.")


if __name__ == "__main__":
    main()
