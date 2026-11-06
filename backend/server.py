"""Serveur local de reconnaissance des signes, pour l app.

Deux reconnaissances, chacune executant TEL QUEL le pipeline de sa demo
Python : meme code, donc meme precision, et aucun pretraitement duplique
dans l app (avec le risque que le modele recoive autre chose que ce qu il a
appris).

  POST /letter   une IMAGE (JPEG ou PNG) -> la lettre.
                 Pipeline de model/src/alphabet/webcam_demo_alphabet.py :
                 miroir, main trouvee par MediaPipe et detouree sur fond blanc,
                 modele fusion image + points de la main (0.9277).
  POST /word     une VIDEO de ~4 s (MP4, WebM...) -> les 5 mots les plus
                 probables. Pipeline de model/src/core/webcam_demo_rtm.py :
                 RTMPose image par image, filtrage des mains, grille a 30 fps,
                 80 images centrees sur le mouvement, ensemble de modeles
                 (0.861 sur 100 signes, signeurs jamais vus).
  GET /health    ce qui est charge.

    python backend/server.py                     # http://127.0.0.1:8000
    python backend/server.py --no-words          # lettres seulement

Le modele des mots a besoin du GPU (onnxruntime-gpu, TensorRT) : s il ne se
charge pas, le serveur sert quand meme les lettres et /word repond 503.

Bibliotheque standard seulement (http.server) : rien a installer en plus des
dependances de model/. Un `pip install` de plus avait deja remplace
onnxruntime-gpu par sa version CPU sans prevenir (model/docs/ASL_CITIZEN.md).
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(os.path.dirname(HERE), "model")
sys.path.insert(0, os.path.join(MODEL, "src", "alphabet"))

import webcam_demo_alphabet as letter_demo  # noqa: E402
from hand_segmenter import GRABCUT_REALTIME, HandSegmenter  # noqa: E402

TOP_LETTERS = 3
TOP_WORDS = 5
# Comme l extraction d ASL Citizen : une video ou la personne manque dans
# plus de 70% des images ne donne pas une sequence exploitable.
MIN_PERSON_RATIO = 0.3
MIN_FRAMES = 8


class LetterRecognizer:
    """Le pipeline de la demo des lettres, une image a la fois.

    MediaPipe tourne en mode VIDEO (suivi d une image a l autre, horodatage
    interne) : les images doivent passer une par une, dans l ordre. D ou le
    verrou, le serveur pouvant recevoir deux requetes en meme temps.
    """

    def __init__(self):
        self.predict, self.classes, self.description = letter_demo.load_classifier()
        self.segmenter = HandSegmenter(num_hands=1, **GRABCUT_REALTIME)
        self.lock = threading.Lock()

    def recognize(self, frame_bgr, mirror=True):
        if mirror:
            # Comme la demo, qui retourne la webcam pour un effet miroir. Le
            # modele a appris avec des miroirs, donc les deux mains passent.
            frame_bgr = cv2.flip(frame_bgr, 1)
        with self.lock:
            cleaned, bbox, found = self.segmenter.process(frame_bgr)
            landmarks = self.segmenter.last_landmarks
            if not found or bbox is None:
                return {"hand": False}
            crop = self.segmenter.crop_for_model(cleaned, bbox,
                                                 output_size=letter_demo.IMG_SIZE)
            if crop is None:
                return {"hand": False}
            probabilities = np.asarray(self.predict(crop, landmarks)).reshape(-1)
        order = np.argsort(probabilities)[::-1][:TOP_LETTERS]
        return {
            "hand": True,
            "letter": self.classes[order[0]],
            "confidence": round(float(probabilities[order[0]]), 4),
            "top": [[self.classes[i], round(float(probabilities[i]), 4)] for i in order],
        }

    def close(self):
        self.segmenter.close()


class WordRecognizer:
    """Le pipeline de la demo des mots, sur une video entiere.

    La demo horodate chaque image a la capture ; ici l horodatage vient du
    decodeur (CAP_PROP_POS_MSEC), comme a l extraction d ASL Citizen : une
    video enregistree par une app a souvent une cadence variable, que
    numero / fps ne decrirait pas. Les images ne sont PAS retournees : la demo
    calcule ses features sur l image brute et ne la retourne que pour
    l affichage.
    """

    def __init__(self):
        sys.path.insert(0, os.path.join(MODEL, "src", "core"))
        import webcam_demo_rtm as word_demo

        self.demo = word_demo
        self.predict, self.meta, layout, confidence = word_demo.load_model()
        self.classes = self.meta["classes"]
        self.centred = bool(self.meta.get("timebase"))
        # Un modele appris sur des features TensorRT doit voir des features
        # TensorRT : sur CUDA, 4.4% des images different.
        tensorrt = self.meta.get("detector", "cuda") == "tensorrt"
        self.source = word_demo.Source(
            with_confidence=confidence, layout=layout, tensorrt=tensorrt,
            gate={key: self.meta.get(key) for key in
                  ("raw_features", "gate", "gate_inside", "gate_conflict",
                   "gate_soft", "extraction_threshold")})
        detector = self.source.model.det_model.session.get_providers()[0]
        expected = (self.meta.get("expected_accuracy") or {}).get("mean")
        self.description = (f"{self.meta['num_classes']} signes, precision attendue "
                            f"{expected}, pose {self.source.provider}, detecteur {detector}")
        self.lock = threading.Lock()

    def recognize(self, video_path):
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        samples, people, index = [], 0, 0
        started = time.perf_counter()
        try:
            with self.lock:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if stamp <= 0.0 and index > 0 and fps > 0:
                        stamp = index / fps
                    vector, _, person = self.source.process(frame)
                    samples.append((stamp, vector))
                    people += bool(person)
                    index += 1
        finally:
            capture.release()

        result = {
            "frames": len(samples),
            "seconds": round(samples[-1][0] - samples[0][0], 2) if samples else 0.0,
            "person": bool(samples) and people / len(samples) >= MIN_PERSON_RATIO,
        }
        if len(samples) < MIN_FRAMES or not result["person"]:
            return result
        top = self.demo.classify(self.predict, samples, self.classes, self.centred)
        result["word"], result["confidence"] = top[0][0], round(top[0][1], 4)
        result["top"] = [[word, round(score, 4)] for word, score in top[:TOP_WORDS]]
        result["processing_seconds"] = round(time.perf_counter() - started, 2)
        return result


def make_handler(letters, words, words_error):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            # L app Flutter en version web appelle depuis une autre origine.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if urlparse(self.path).path != "/health":
                self._send(404, {"error": "inconnu ; voir POST /letter, POST /word"})
                return
            self._send(200, {
                "ok": True,
                "letters": letters.description,
                "words": words.description if words else None,
                "words_error": words_error,
            })

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/letter":
                data = np.frombuffer(self._body(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                if frame is None:
                    self._send(400, {"error": "corps attendu : une image JPEG ou PNG"})
                    return
                mirror = parse_qs(url.query).get("mirror", ["1"])[0] != "0"
                self._send(200, letters.recognize(frame, mirror=mirror))
            elif url.path == "/word":
                if words is None:
                    self._send(503, {"error": f"modele des mots non charge : {words_error}"})
                    return
                body = self._body()
                if not body:
                    self._send(400, {"error": "corps attendu : une video"})
                    return
                # OpenCV ne decode une video que depuis un fichier.
                handle, path = tempfile.mkstemp(suffix=".mp4")
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write(body)
                    result = words.recognize(path)
                finally:
                    os.remove(path)
                if result is None:
                    self._send(400, {"error": "video illisible (MP4 ou WebM attendu)"})
                else:
                    self._send(200, result)
            else:
                self._send(404, {"error": "inconnu ; voir POST /letter, POST /word"})

        def log_message(self, format, *args):
            pass                     # une ligne par image noierait la console

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 pour un telephone sur le meme reseau ; "
                             "le serveur est alors joignable par tout ce reseau")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-words", action="store_true",
                        help="ne pas charger le modele des mots (GPU, ~1 min)")
    args = parser.parse_args()

    letters = LetterRecognizer()
    print(f"Lettres : {letters.description}")
    words, words_error = None, "desactive (--no-words)" if args.no_words else None
    if not args.no_words:
        try:
            words = WordRecognizer()
            print(f"Mots : {words.description}")
        except (Exception, SystemExit) as error:
            words_error = f"{type(error).__name__}: {error}"
            print(f"Mots : NON CHARGES ({words_error}) ; /word repondra 503")

    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(letters, words, words_error))
    print(f"Pret : http://{args.host}:{args.port}  (POST /letter, POST /word ; "
          f"Ctrl+C pour arreter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        letters.close()


if __name__ == "__main__":
    main()
