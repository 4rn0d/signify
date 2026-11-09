# Backend

## Installation (coequipiers)

Sans le serveur, l app marche quand meme : la page d entrainement affiche
« Letter server unreachable » et le bouton « Done » valide a la main. Le
serveur ajoute la reconnaissance automatique.

**Lettres** (Windows, macOS ou Linux, pas de GPU necessaire). Le modele des
lettres est dans le depot ; le detecteur de main de MediaPipe se telecharge
tout seul au premier lancement (8 Mo).

```
python -m venv .venv
.venv\Scripts\activate                  # macOS / Linux : source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/server.py --no-words
```

**L app** demande Flutter 3.47 ou plus recent (Dart 3.13.5) ; sous Windows,
activer le mode developpeur (`start ms-settings:developers`) sinon les plugins
ne compilent pas.

```
cd frontend
flutter pub get
flutter run -d windows                  # ou macos
```

**Mots** (optionnel ; Windows ou Linux avec GPU NVIDIA). Les poids du modele
des mots ne sont PAS dans le depot : le depot est public, et ce modele est
entraine sur ASL Citizen (licence Microsoft Research : recherche non
commerciale, redistribution interdite). Ils se partagent en prive :

1. demander `aslc_word_model.zip` a Ulrich (Teams, cle USB... jamais sur
   GitHub ni sur un lien public) ;
2. le dezipper dans `model/models/` (5 fichiers `aslc_gru_rtmpose_s0_k*.keras`) ;
3. `pip install -r model/requirements.txt` (RTMPose, onnxruntime-gpu,
   TensorRT), puis les deux lignes de reparation onnxruntime du
   `model/README.md` ;
4. `python backend/server.py` (sans `--no-words`).

Si les poids manquent, le serveur le dit au demarrage et sert quand meme les
lettres. Pour refaire l archive apres un nouvel entrainement :
`Compress-Archive model/models/aslc_gru_rtmpose_s*.keras aslc_word_model.zip`.

## Serveur des signes (`server.py`)

Un serveur local qui reconnait les lettres et les mots pour l app. Chaque
reconnaissance execute exactement le pipeline de sa demo Python, donc avec la
meme precision. Il utilise l environnement Python de `model/` (voir
`model/requirements.txt`) et n ajoute aucune dependance.

```
python backend/server.py              # lettres + mots, http://127.0.0.1:8000
python backend/server.py --no-words   # lettres seulement (pas besoin de GPU)
```

Le modele des mots a besoin du GPU (onnxruntime-gpu, TensorRT) et met environ
une minute a charger. S il ne se charge pas, le serveur sert quand meme les
lettres et `/word` repond 503 ; `/health` dit pourquoi.

### Lettres : `POST /letter`

La page d entrainement de l app (`frontend/lib/pages/training_page.dart`)
l utilise deja. Camera allumee, l app envoie une image toutes les 0.4 s ; une
lettre est validee quand le modele la voit sur 3 images de suite a 80 % de
confiance ou plus. Le bouton « Done » reste disponible.

Pipeline de `model/src/alphabet/webcam_demo_alphabet.py` : main trouvee par
MediaPipe, detouree sur fond blanc, modele fusion image + points de la main
(0.9277 en validation).

```
POST /letter     corps : une image JPEG ou PNG ; ?mirror=0 pour ne pas la retourner
  -> {"hand": true, "letter": "B", "confidence": 0.91, "top": [["B", 0.91], ...]}
  -> {"hand": false}                     aucune main vue
```

Sur les 26 photos de test du jeu ASL Alphabet (une par lettre) : 19 justes, 3
erreurs (R, S, Z), 3 photos ou MediaPipe ne trouve aucune main (H, N, O ; la
demo a la meme limite). Photos isolees de 200x200 ; devant une webcam, la main
suivie d une image a l autre est mieux detectee.

### Mots : `POST /word`

Une video d environ 4 s (MP4, WebM...) d une personne qui fait UN signe ; la
reponse donne les 5 mots les plus probables parmi les 100 du vocabulaire
(liste : `model/models/aslc_gru_rtmpose.meta.json`, cle `classes`).

Pipeline de `model/src/core/webcam_demo_rtm.py` : RTMPose image par image,
filtrage des mains, grille a 30 images/s, 80 images centrees sur le mouvement,
ensemble de 5 modeles. Precision attendue 0.861 sur des signeurs jamais vus
(validation croisee, `model/models/aslc_kfold_baseline.json`). Le moment du
signe dans la video importe peu : la fenetre se centre sur le mouvement.

```
POST /word       corps : la video
  -> {"person": true, "word": "HELLO", "confidence": 0.97,
      "top": [["HELLO", 0.97], ...], "frames": 54, "seconds": 2.12,
      "processing_seconds": 1.18}
  -> {"person": false, "frames": ..., "seconds": ...}   personne visible dans
                                                         moins de 30 % des images
```

Teste sur 12 videos ASL Citizen : 12 justes, 1 a 2 s de traitement chacune.
Ces videos ont servi a l entrainement du modele final : le test verifie la
chaine, pas la precision.

### Autres appareils

| ou tourne l app | serveur | app |
|---|---|---|
| meme ordinateur | `python backend/server.py` | rien a changer |
| emulateur Android | idem | `flutter run --dart-define=LETTER_SERVER=http://10.0.2.2:8000` |
| telephone sur le meme Wi-Fi | `python backend/server.py --host 0.0.0.0` | `--dart-define=LETTER_SERVER=http://<ip-du-pc>:8000` |

Avec `--host 0.0.0.0`, tout le reseau local peut joindre le serveur.

### Licence

Le modele des mots est entraine sur ASL Citizen (licence Microsoft Research :
recherche non commerciale). Projet etudiant de recherche, sans diffusion au
public : a garder ainsi, et a ne pas deployer comme service public.
