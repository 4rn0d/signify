"""Telecharge les videos WLASL d'un sous-ensemble.

Constats du probe (voir docs/WLASL.md) :

- 4 hebergeurs sont morts et representent 669 clips sur 2038 : signingsavvy
  (403), handspeak (404), aslpro (404, des .swf Flash de toute facon) et
  aslsignbank (DNS). On ne les tente pas : 669 echecs certains, c'est du
  temps et du trafic pour rien.
- YouTube represente 34% du sous-ensemble et environ 70% y survit.
- Mediane de duree : 3 secondes.

Resolution : on n'extrait que des landmarks, pas des pixels. Une video en
480p suffit largement a MediaPipe et pese une fraction d'une HD. On demande
donc le plus petit format >= MAX_HEIGHT, sans audio.

Reprise : un fichier deja present est saute. Le manifeste est reecrit a
chaque run, donc relancer le script met a jour l'etat sans retelecharger.
"""

import argparse
import collections
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
METADATA_PATH = os.path.join(WLASL_DIR, "WLASL_v0.3.json")
VIDEO_DIR = os.path.join(WLASL_DIR, "videos")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")

# Verifies un par un : 403, 404, 404, echec DNS.
DEAD_HOSTS = {
    "www.signingsavvy.com",          # 403, bloque l'acces automatise
    "www.handspeak.com",             # 404
    "www.aslpro.com",                # 404 (des .swf Flash de toute facon)
    "aslsignbank.haskins.yale.edu",  # echec DNS
    # Domaine parque : repond 200 avec une page HTML de redirection vers
    # /lander, pas une video. Un code 200 ne garantit donc rien — c'est le
    # controle de taille minimale ci-dessous qui l'a demasque.
    "www.aslsearch.com",
}

MAX_HEIGHT = 480
TIMEOUT = 45

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class _SilentLogger:
    """yt-dlp ecrit directement sinon ; on garde la sortie du script lisible."""

    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def host_of(url):
    return urllib.parse.urlparse(url).netloc


def is_youtube(url):
    return "youtu" in host_of(url)


def load_subset(size, metadata_path=METADATA_PATH):
    with open(metadata_path, encoding="utf-8") as handle:
        data = json.load(handle)

    ranked = sorted(data, key=lambda g: len(g["instances"]), reverse=True)

    clips = []
    for gloss in ranked[:size]:
        for inst in gloss["instances"]:
            clips.append(
                {
                    "video_id": inst["video_id"],
                    "gloss": gloss["gloss"],
                    "url": inst["url"],
                    "signer_id": inst["signer_id"],
                    # Sans lui, --variants dominant ne filtre plus rien :
                    # 19 des 50 gloses cibles ont plusieurs variantes, et
                    # n en garder qu une valait +5.6 points.
                    "variation_id": inst.get("variation_id", 0),
                    "split": inst["split"],
                    "fps": inst["fps"],
                    "frame_start": inst["frame_start"],
                    "frame_end": inst["frame_end"],
                    "bbox": inst["bbox"],
                }
            )
    return clips


def target_path(video_id):
    return os.path.join(VIDEO_DIR, f"{video_id}.mp4")


def download_direct(clip):
    """Telechargement HTTP simple pour les hebergeurs non-YouTube."""
    path = target_path(clip["video_id"])

    request = urllib.request.Request(
        clip["url"],
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"https://{host_of(clip['url'])}/",
        },
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = response.read()

    if len(data) < 2048:
        raise ValueError(f"reponse trop courte ({len(data)} octets)")

    # Ecriture atomique : un fichier partiel serait saute comme "deja la".
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)

    return len(data)


def download_youtube(clip):
    import yt_dlp

    path = target_path(clip["video_id"])

    options = {
        "quiet": True,
        "no_warnings": True,
        # quiet ne suffit pas : la barre de progression et les erreurs
        # partent quand meme sur stdout/stderr et noient la sortie.
        "noprogress": True,
        "logger": _SilentLogger(),
        "noplaylist": True,
        "socket_timeout": TIMEOUT,
        "outtmpl": path,
        # Plus petit format d'au moins MAX_HEIGHT, sinon le meilleur
        # disponible. Pas d'audio : inutile et ca evite d'avoir besoin de
        # ffmpeg pour remuxer.
        "format": (
            f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]"
            f"/best[height<={MAX_HEIGHT}][ext=mp4]"
            f"/best[ext=mp4]/best"
        ),
        "retries": 2,
        "fragment_retries": 2,
        # Le client web se fait refuser par la detection de bots de YouTube
        # ("Sign in to confirm you're not a bot") : 285 echecs sur ce lot.
        # Le client android passe sans authentification — teste 6/6 contre
        # 0/6 pour web, ios et tv.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([clip["url"]])

    if not os.path.exists(path):
        raise FileNotFoundError("yt-dlp n'a produit aucun fichier")

    return os.path.getsize(path)


SAFE = re.compile(r"[^a-z0-9_]")


def organised_path(clip):
    """Ou organise_videos.py aurait range ce clip.

    Doit rester identique a target_name() dans organise_videos.py : meme
    nettoyage du nom de glose, meme motif de fichier.
    """
    gloss = SAFE.sub("", clip["gloss"].replace(" ", "_").lower())
    return os.path.join(
        VIDEO_DIR, gloss, f"{gloss}_s{clip['signer_id']}_{clip['video_id']}.mp4")


def existing_file(clip):
    """Le clip est-il deja sur le disque, a plat OU range ?"""
    for path in (target_path(clip["video_id"]), organised_path(clip)):
        if os.path.exists(path) and os.path.getsize(path) > 2048:
            return path
    return None


def fetch(clip):
    """Retourne (clip, status, detail)."""
    path = target_path(clip["video_id"])

    # Le cache doit regarder les DEUX dispositions : relancer sur un dossier
    # deja range retenterait sinon tous les clips existants, et ceux dont le
    # lien est mort repasseraient en "failed" dans le manifeste reecrit.
    found = existing_file(clip)
    if found is not None:
        return clip, "cached", os.path.getsize(found)

    if host_of(clip["url"]) in DEAD_HOSTS:
        return clip, "skipped_dead_host", host_of(clip["url"])

    try:
        if is_youtube(clip["url"]):
            size = download_youtube(clip)
        else:
            size = download_direct(clip)
        return clip, "ok", size
    except Exception as error:
        detail = f"{type(error).__name__}: {str(error)[:120]}"
        return clip, "failed", detail


def merge_manifest(fresh, path=MANIFEST_PATH):
    """Fusionne les nouveaux resultats dans le manifeste existant.

    Deux regles, et la seconde compte autant que la premiere :

      - les entrees absentes des nouveaux resultats sont CONSERVEES. Un run
        partiel ou limite ne doit pas effacer le reste du dataset.
      - une entree qui possede un fichier n est jamais degradee par un
        resultat sans fichier. Un lien mort aujourd hui ne doit pas effacer un
        clip recupere hier par le miroir.
    """
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for record in json.load(handle):
                existing[record["video_id"]] = record

    kept = replaced = protected = 0
    for record in fresh:
        old_record = existing.get(record["video_id"])
        if old_record is not None and old_record.get("path") and not record.get("path"):
            protected += 1          # on garde le clip qu on a deja
            continue
        if old_record is None:
            kept += 1
            existing[record["video_id"]] = record
        else:
            replaced += 1
            # Champ par champ, et non remplacement : un champ absent du
            # nouveau resultat est conserve de l ancien. C est ce qui a
            # failli effacer variation_id de 1321 entrees, et ce qui
            # protegera le prochain champ que ce script oubliera.
            merged_record = dict(old_record)
            merged_record.update(record)
            existing[record["video_id"]] = merged_record

    merged = sorted(existing.values(), key=lambda r: str(r["video_id"]))
    return merged, {"ajoutes": kept, "mis_a_jour": replaced,
                    "preserves": protected, "total": len(merged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4,
                        help="volontairement bas : ce sont de petits sites educatifs")
    parser.add_argument("--limit", type=int, default=0,
                        help="ne traiter que les N premiers clips (test)")
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    clips = load_subset(args.subset)
    if args.limit:
        clips = clips[: args.limit]

    attempted = [c for c in clips if host_of(c["url"]) not in DEAD_HOSTS]
    print(f"WLASL{args.subset}: {len(clips)} clips")
    print(f"  {len(clips) - len(attempted)} ignores (hebergeurs morts)")
    print(f"  {len(attempted)} a tenter, {args.workers} connexions\n")

    results = []
    counts = collections.Counter()
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, c): c for c in clips}

        for index, future in enumerate(as_completed(futures), start=1):
            clip, status, detail = future.result()
            counts[status] += 1
            results.append(
                {
                    "video_id": clip["video_id"],
                    "gloss": clip["gloss"],
                    "signer_id": clip["signer_id"],
                    "variation_id": clip.get("variation_id", 0),
                    "split": clip["split"],
                    "fps": clip["fps"],
                    "frame_start": clip["frame_start"],
                    "frame_end": clip["frame_end"],
                    "bbox": clip["bbox"],
                    "url": clip["url"],
                    "status": status,
                    "detail": detail if status == "failed" else None,
                    "path": os.path.relpath(target_path(clip["video_id"]), PROJECT_ROOT)
                    if status in ("ok", "cached")
                    else None,
                }
            )

            if index % 50 == 0 or index == len(clips):
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-6)
                remaining = (len(clips) - index) / max(rate, 1e-6)
                print(
                    f"  {index}/{len(clips)}  "
                    f"ok {counts['ok']}  cache {counts['cached']}  "
                    f"echec {counts['failed']}  ignore {counts['skipped_dead_host']}"
                    f"   ~{remaining / 60:.0f} min restantes",
                    flush=True,
                )

    merged, stats = merge_manifest(results)

    if args.limit:
        # --limit est documente "pour test" et reecrivait pourtant le
        # manifeste : il est passe de 2038 entrees a 200. Un run limite ne
        # traite qu un echantillon, donc il n a rien a dire sur le reste.
        print()
        print(f"  --limit actif : manifeste NON ecrit "
              f"({stats['ajoutes']} auraient ete ajoutes, "
              f"{stats['mis_a_jour']} mis a jour)")
    else:
        temporary = MANIFEST_PATH + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=1)
        os.replace(temporary, MANIFEST_PATH)    # remplacement atomique
        print()
        print(f"  manifeste fusionne : {stats['total']} entrees "
              f"({stats['ajoutes']} ajoutees, {stats['mis_a_jour']} mises a jour, "
              f"{stats['preserves']} preservees d un clip deja possede)")

    have = counts["ok"] + counts["cached"]
    print(f"\n{'=' * 60}")
    print(f"  recuperees : {have}/{len(clips)} ({have / len(clips):.0%})")
    print(f"  echecs     : {counts['failed']}")
    print(f"  ignorees   : {counts['skipped_dead_host']} (hebergeurs morts)")
    print(f"  manifeste  : "
          + ("NON ecrit (--limit)" if args.limit else MANIFEST_PATH))
    print(f"\n  Lance audit_downloaded.py pour voir ce qui reste par glose.")


if __name__ == "__main__":
    main()
