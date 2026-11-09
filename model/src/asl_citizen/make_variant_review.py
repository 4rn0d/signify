"""Page locale pour apparier les signes d ASL Citizen aux variantes apprises.

ASL Citizen eclate un mot anglais en signes numerotes quand il en recouvre
plusieurs (CANDY1, CANDY2), chacun avec son code ASL-LEX. Mais sa
numerotation n a aucune raison de suivre celle de WLASL : pour `candy`, le
modele a appris la variante 1 de WLASL (14 clips contre 10), et rien ne dit
si c est CANDY1 ou CANDY2. Ajouter les clips du mauvais signe reintroduirait
les etiquettes contradictoires que `--variants dominant` a retirees (+5.6).

Le seul juge fiable est l oeil : la page met chaque clip de reference WLASL
(ceux de make_reference.py, ceux que tes signeurs imitent) a cote de deux
clips de chaque signe candidat, et tu choisis. L export donne
variant_map.json, que l extraction lira.

PAGE LOCALE, JAMAIS PUBLIEE : la licence d ASL Citizen interdit de
redistribuer les videos et d en faire un service heberge. La page est ecrite
dans data/asl_citizen/, qui est gitignore, et lit les videos sur le disque.

    python src/asl_citizen/make_variant_review.py
    puis ouvrir data/asl_citizen/variant_review.html
"""

import collections
import csv
import html
import json
import os
import random
import re

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIZEN = os.path.join(ROOT, "data", "asl_citizen", "ASL_Citizen")
REFERENCE = os.path.join(ROOT, "data", "wlasl", "reference")
CARD = os.path.join(ROOT, "models", "wlasl", "wlasl_gru_rtmpose.meta.json")
MANIFEST = os.path.join(ROOT, "data", "wlasl", "download_manifest.json")
OUTPUT = os.path.join(ROOT, "data", "asl_citizen", "variant_review.html")

SAMPLES = 2


def load_rows():
    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(CITIZEN, "splits", f"{split}.csv")
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["split"] = split
                rows.append(row)
    return rows


def playable(path):
    """Un navigateur lit le h264, pas le FMP4 (MPEG-4 partie 2)."""
    capture = cv2.VideoCapture(path)
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) in ("h264", "avc1")


def wlasl_variants(glosses):
    counts = collections.defaultdict(collections.Counter)
    with open(MANIFEST, encoding="utf-8") as handle:
        for record in json.load(handle):
            if record.get("gloss") in glosses:
                counts[record["gloss"]][record.get("variation_id", 0)] += 1
    return counts


def main():
    with open(CARD, encoding="utf-8") as handle:
        ours = json.load(handle)["classes"]

    rows = load_rows()
    by_gloss = collections.defaultdict(list)
    codes = {}
    for row in rows:
        by_gloss[row["Gloss"]].append(row)
        codes[row["Gloss"]] = row["ASL-LEX Code"]

    variants = wlasl_variants(set(ours))
    rng = random.Random(0)

    entries = []
    for gloss in ours:
        key = gloss.upper().replace(" ", "_")
        family = sorted(g for g in by_gloss
                        if re.fullmatch(re.escape(key) + r"\d*", g))
        wlasl = variants.get(gloss, collections.Counter())
        dominant = max(wlasl, key=wlasl.get) if wlasl else 0
        if len(family) > 1:
            priority, reason = 0, f"eclatee en {len(family)} signes dans ASL Citizen"
        elif len(wlasl) > 1:
            priority, reason = 1, (f"2 variantes dans WLASL "
                                   f"({dict(sorted(wlasl.items()))}, apprise : {dominant})")
        elif family:
            priority, reason = 2, "un seul signe des deux cotes"
        else:
            priority, reason = 3, "absente d ASL Citizen"

        candidates = []
        for name in family:
            pool = [r for r in by_gloss[name] if r["split"] == "train"] or by_gloss[name]
            rng.shuffle(pool)
            chosen, signers = [], set()
            for row in pool:
                path = os.path.join(CITIZEN, "videos", row["Video file"])
                if row["Participant ID"] in signers or not os.path.exists(path):
                    continue
                if not playable(path):
                    continue
                chosen.append(os.path.relpath(path, os.path.dirname(OUTPUT)))
                signers.add(row["Participant ID"])
                if len(chosen) == SAMPLES:
                    break
            candidates.append({"name": name, "code": codes[name],
                               "clips": len(by_gloss[name]), "videos": chosen})

        references = [
            os.path.relpath(os.path.join(REFERENCE, f"{gloss}_{i}.mp4"),
                            os.path.dirname(OUTPUT))
            for i in range(1, 3)
            if os.path.exists(os.path.join(REFERENCE, f"{gloss}_{i}.mp4"))
        ]
        entries.append({"gloss": gloss, "priority": priority, "reason": reason,
                        "references": references, "candidates": candidates})

    entries.sort(key=lambda e: (e["priority"], e["gloss"]))
    write_page(entries)
    counts = collections.Counter(e["priority"] for e in entries)
    print(f"  {len(entries)} gloses : {counts[0]} eclatees, {counts[1]} a 2 variantes "
          f"WLASL, {counts[2]} simples, {counts[3]} absentes")
    print(f"  ouvrir {OUTPUT}")


def video(src, label):
    return (f'<figure><video src="{html.escape(src)}" autoplay muted loop '
            f'playsinline preload="metadata"></video>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')


def write_page(entries):
    sections = []
    for entry in entries:
        gloss = entry["gloss"]
        refs = "".join(video(src, f"reference {i + 1}")
                       for i, src in enumerate(entry["references"]))
        options = []
        for candidate in entry["candidates"]:
            clips = "".join(video(src, f"{candidate['name']} #{i + 1}")
                            for i, src in enumerate(candidate["videos"]))
            options.append(
                f'<label class="option"><input type="radio" name="{gloss}" '
                f'value="{candidate["name"]}"><span class="name">{candidate["name"]}'
                f'</span><span class="meta">{candidate["code"]} &middot; '
                f'{candidate["clips"]} clips</span><div class="clips">{clips}</div>'
                f'</label>')
        if entry["candidates"]:
            options.append(
                f'<label class="option none"><input type="radio" name="{gloss}" '
                f'value=""><span class="name">aucun ne correspond</span></label>')
        body = "".join(options) or '<p class="absent">rien a apparier</p>'
        sections.append(
            f'<section data-priority="{entry["priority"]}"><header><h2>{gloss}</h2>'
            f'<span class="reason">{html.escape(entry["reason"])}</span></header>'
            f'<div class="row"><div class="refs"><h3>Ce que le modele a appris'
            f'</h3>{refs}</div><div class="candidates"><h3>ASL Citizen</h3>{body}'
            f'</div></div></section>')

    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Appariement des variantes</title>
<style>
:root {{ --bg:#f7f6f3; --fg:#1d1d1b; --muted:#6b6a66; --line:#dddbd5;
        --card:#ffffff; --accent:#2f6b4f; --pick:#e3efe8; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161615; --fg:#ecebe7; --muted:#9a9893; --line:#33322f;
          --card:#1f1f1d; --accent:#7cc39c; --pick:#22352b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.45 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 120px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
.lead {{ color:var(--muted); margin:0 0 24px; max-width:70ch; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:16px; }}
section header {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }}
h2 {{ margin:0; font-size:18px; }}
h3 {{ margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.06em;
      color:var(--muted); font-weight:600; }}
.reason {{ color:var(--muted); font-size:13px; }}
.row {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,2fr); gap:20px;
        margin-top:12px; }}
@media (max-width:760px) {{ .row {{ grid-template-columns:1fr; }} }}
.refs, .clips {{ display:flex; gap:8px; flex-wrap:wrap; }}
figure {{ margin:0; }}
video {{ width:220px; max-width:100%; border-radius:6px; background:#000; display:block; }}
figcaption {{ font-size:12px; color:var(--muted); margin-top:2px; }}
.option {{ display:block; border:1px solid var(--line); border-radius:8px; padding:10px;
          margin-bottom:8px; cursor:pointer; }}
.option:has(input:checked) {{ background:var(--pick); border-color:var(--accent); }}
.option .name {{ font-weight:600; margin:0 8px 0 6px; }}
.option .meta {{ color:var(--muted); font-size:13px; }}
.option .clips {{ margin-top:8px; }}
.absent {{ color:var(--muted); margin:0; }}
.bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--card);
        border-top:1px solid var(--line); padding:12px 16px; display:flex; gap:12px;
        align-items:center; justify-content:center; flex-wrap:wrap; }}
button {{ font:inherit; padding:8px 16px; border-radius:6px; border:0;
          background:var(--accent); color:var(--bg); cursor:pointer; }}
#count {{ color:var(--muted); }}
</style></head><body><main>
<h1>Appariement des variantes</h1>
<p class="lead">Pour chaque mot, choisis le signe d ASL Citizen qui correspond a la
reference de gauche, c est-a-dire ce que le modele a appris et ce que tes
signeurs imitent. En haut, les cas ou le choix compte vraiment. Page locale : les
videos d ASL Citizen ne peuvent etre ni publiees ni partagees.</p>
{"".join(sections)}
</main>
<div class="bar"><span id="count"></span>
<button id="export">Exporter variant_map.json</button></div>
<script>
const KEY = "variant-review";
const names = [...new Set([...document.querySelectorAll("input[type=radio]")].map(i => i.name))];
function state() {{
  const out = {{}};
  for (const n of names) {{
    const c = document.querySelector(`input[name="${{CSS.escape(n)}}"]:checked`);
    if (c) out[n] = c.value || null;
  }}
  return out;
}}
function refresh() {{
  const s = state();
  document.getElementById("count").textContent =
    `${{Object.keys(s).length}} / ${{names.length}} gloses tranchees`;
  try {{ localStorage.setItem(KEY, JSON.stringify(s)); }} catch (e) {{}}
}}
try {{
  const saved = JSON.parse(localStorage.getItem(KEY) || "{{}}");
  for (const [n, v] of Object.entries(saved)) {{
    const el = document.querySelector(
      `input[name="${{CSS.escape(n)}}"][value="${{CSS.escape(v || "")}}"]`);
    if (el) el.checked = true;
  }}
}} catch (e) {{}}
document.addEventListener("change", refresh);
document.getElementById("export").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify(state(), null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "variant_map.json";
  a.click();
}});
refresh();
</script></body></html>"""

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(page)


if __name__ == "__main__":
    main()
