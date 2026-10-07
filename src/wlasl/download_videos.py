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


def fetch(clip):
    """Retourne (clip, status, detail)."""
    path = target_path(clip["video_id"])

    if os.path.exists(path) and os.path.getsize(path) > 2048:
        return clip, "cached", os.path.getsize(path)

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

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)

    have = counts["ok"] + counts["cached"]
    print(f"\n{'=' * 60}")
    print(f"  recuperees : {have}/{len(clips)} ({have / len(clips):.0%})")
    print(f"  echecs     : {counts['failed']}")
    print(f"  ignorees   : {counts['skipped_dead_host']} (hebergeurs morts)")
    print(f"  manifeste  : {MANIFEST_PATH}")
    print(f"\n  Lance audit_downloaded.py pour voir ce qui reste par glose.")


if __name__ == "__main__":
    main()
