"""Page locale des signes cibles : comment faire chacun, pour tester la demo.

Pour chaque signe du vocabulaire (rang ASL-LEX), deux clips ASL Citizen de
signeurs differents, le sens anglais et les indices de phonologie d ASL-LEX
(forme de main, emplacement, mouvement). Les clips montrent exactement la
variante que le modele a apprise, ce qui compte pour les signes numerotes
(CANDY1 et CANDY2 sont deux signes differents).

PAGE LOCALE, JAMAIS PUBLIEE : la licence d ASL Citizen interdit de
redistribuer les videos et d en faire un service heberge. La page est ecrite
dans data/asl_citizen/, qui est gitignore, et lit les videos sur le disque.
Elle ne sert pas non plus de consigne pour d autres signeurs : leur envoyer
ces clips serait les redistribuer.

    python src/asl_citizen/make_sign_sheet.py            # les 100 cibles
    puis ouvrir data/asl_citizen/target_signs.html
"""

import argparse
import collections
import csv
import html
import json
import os
import random

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIZEN = os.path.join(ROOT, "data", "asl_citizen", "ASL_Citizen")
RANKED = os.path.join(ROOT, "data", "asl_citizen", "ranked_signs.json")
ASL_LEX = os.path.join(ROOT, "data", "asl_lex", "signdata.csv")
OUTPUT = os.path.join(ROOT, "data", "asl_citizen", "target_signs.html")

HINTS = (("Handshape.2.0", "forme"), ("MajorLocation.2.0", "emplacement"),
         ("Movement.2.0", "mouvement"), ("SignType.2.0", "mains"))


def playable(path):
    """Un navigateur lit le h264, pas le FMP4 (MPEG-4 partie 2)."""
    capture = cv2.VideoCapture(path)
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) in ("h264", "avc1")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    with open(RANKED, encoding="utf-8") as handle:
        signs = json.load(handle)[:args.top]
    # signdata.csv n est pas en UTF-8 valide partout ; latin-1 lit tout.
    with open(ASL_LEX, encoding="latin-1") as handle:
        lex = {row["Code"]: row for row in csv.DictReader(handle)}

    clips = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        with open(os.path.join(CITIZEN, "splits", f"{split}.csv"), encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                clips[row["Gloss"]].append(row)

    picker = random.Random(0)
    cards = []
    for rank, sign in enumerate(signs, 1):
        rows = clips[sign["gloss"]][:]
        picker.shuffle(rows)
        chosen, signers = [], set()
        for row in rows:
            path = os.path.join(CITIZEN, "videos", row["Video file"])
            if row["Participant ID"] in signers or not playable(path):
                continue
            chosen.append(path)
            signers.add(row["Participant ID"])
            if len(chosen) == args.samples:
                break
        entry = lex.get(sign["code"], {})
        meaning = entry.get("SignBankEnglishTranslations") or entry.get("EntryID", "")
        hints = " · ".join(f"{label} {entry[key]}" for key, label in HINTS if entry.get(key))
        videos = "".join(
            f'<video src="{html.escape("file:///" + p.replace(os.sep, "/"))}" '
            f'controls loop muted preload="metadata"></video>' for p in chosen)
        search = html.escape(f'{sign["gloss"]} {meaning}'.lower())
        cards.append(
            f'<section id="s{rank}" data-search="{search}">'
            f'<h2><span>{rank}</span> {html.escape(sign["gloss"])}</h2>'
            f'<p class="meaning">{html.escape(meaning)}</p>'
            f'<p class="hints">{html.escape(hints)}</p>'
            f'<div class="videos">{videos or "<em>aucun clip lisible par le navigateur</em>"}</div>'
            f'</section>')
        print(f"  {rank:3d} {sign['gloss']:14s} {len(chosen)} clips   {meaning[:50]}")

    # Index alphabetique : une grille de 100 cartes par rang ne se parcourt
    # pas quand on cherche un mot precis.
    index = "".join(
        f'<a href="#s{rank}">{html.escape(sign["gloss"])}</a>'
        for rank, sign in sorted(enumerate(signs, 1), key=lambda item: item[1]["gloss"]))

    page = f"""<!doctype html><meta charset="utf-8">
<title>Signes cibles</title>
<style>
:root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e2dc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ececea; --muted: #9a9a94; --line: #2c2c2a; }} }}
body {{ margin: 0 auto; max-width: 1100px; padding: 24px 16px; background: var(--bg); color: var(--fg);
        font: 15px/1.45 system-ui, sans-serif; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
header p {{ color: var(--muted); margin: 0 0 20px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
section {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
h2 {{ font-size: 17px; margin: 0; }}
h2 span {{ color: var(--muted); font-weight: 400; margin-right: 6px; }}
.meaning {{ margin: 2px 0; }}
.hints {{ color: var(--muted); font-size: 13px; margin: 2px 0 8px; }}
.videos {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
video {{ width: 100%; border-radius: 4px; background: #000; }}
.bar {{ position: sticky; top: 0; z-index: 1; background: var(--bg); padding: 10px 0;
        border-bottom: 1px solid var(--line); margin-bottom: 16px; }}
input {{ width: 100%; box-sizing: border-box; font: inherit; padding: 8px 10px; border-radius: 6px;
         border: 1px solid var(--line); background: var(--bg); color: var(--fg); }}
nav {{ display: flex; flex-wrap: wrap; gap: 4px 10px; margin-top: 8px; max-height: 96px; overflow-y: auto;
       font-size: 13px; }}
nav a {{ color: var(--fg); text-decoration: none; }}
nav a:hover {{ text-decoration: underline; }}
section {{ scroll-margin-top: 170px; }}
section:target {{ outline: 2px solid #3b82f6; }}
.hidden {{ display: none; }}
</style>
<header><h1>{len(signs)} signes cibles</h1>
<p>Par rang de frequence ASL-LEX. Clips ASL Citizen, usage local uniquement (licence : ne pas partager).</p></header>
<div class="bar">
<input id="q" type="search" placeholder="Chercher un signe ou son sens anglais (ex. water, eat, thank)..." autofocus>
<nav>{index}</nav>
</div>
<main>{''.join(cards)}</main>
<script>
const q = document.getElementById("q");
q.addEventListener("input", () => {{
  const words = q.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
  for (const card of document.querySelectorAll("section")) {{
    const text = card.dataset.search;
    card.classList.toggle("hidden", !words.every(w => text.includes(w)));
  }}
}});
</script>
"""
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(page)
    print(f"\n  page : {OUTPUT}")


if __name__ == "__main__":
    main()
