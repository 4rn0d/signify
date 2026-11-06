"""Entraine le modele final sur TOUTES les sequences et le sauvegarde.

train_kfold.py mesure la precision attendue mais ne garde aucun poids : son
role est d'estimer, pas de produire un modele. L'estimation faite, on
reentraine sur l'ensemble des donnees — garder un fold de cote ne servirait
qu'a repeter une mesure deja obtenue, en se privant de 20% des exemples.

La configuration est celle mesuree a 0.7413 +- 0.0380 en validation croisee
par signeur, soit 37x le hasard :

  --features-dir data/wlasl/features_rtm_all   RTMPose en mode BRUT
  --variants dominant        une variante de signe par glose (+5.6)
  --rich                     23 scalaires derives de x,y (+5.9)
  --confidence               la confiance de chaque main, en continu (+1.6)
  --handshape-full           26 scalaires de configuration de main
  --project 48               projection lineaire avant le GRU (+4.7)
  --gate 0.7 --gate-inside 0.5 --gate-conflict
  --pretrain-vocab 300 --pretrain-epochs 120 --pretrain-variants all
  --seeds 5 --snapshots 5    25 modeles dans l ensemble

Deux points ou ce script peut silencieusement cesser de correspondre a la
mesure, et les deux ont deja coute cher dans ce projet :

1. LE FILTRAGE. features_rtm_all est extrait en mode BRUT : les deux mains
   sont toujours ecrites, drapeaux a 1, et c est gate_hands() qui ecarte
   ensuite. Le modele n a donc jamais vu de features non filtrees, et la
   demo doit appliquer le MEME gate_hands avec les MEMES parametres — pas
   le filtrage interne de l extracteur, dont la regle de conflit differe
   (distance au poignet contre confiance la plus faible).

2. LE VOCABULAIRE. features_rtm_all contient 300 gloses. Sans --vocab-from,
   les 50 cibles seraient les 50 plus fournies de ces 300, soit un AUTRE
   ensemble que celui mesure, et le chiffre de la fiche ne decrirait plus
   ce modele.

Pas de --no-z : RTMPose ne rend que x,y, il n y a pas de profondeur a
retirer.

La TTA (moyenne avec l'image miroir) est appliquee a l'inference, dans la
demo : elle ne change rien a l'entrainement.

Attention : le score inscrit dans la fiche vient du k-fold, PAS de ce
modele. Un modele evalue sur ses propres donnees d'entrainement afficherait
un chiffre sans signification (0.815 contre 0.37 reels, mesure plus tot).
"""

import argparse
import collections
import datetime
import json
import os

import glob

import numpy as np
import tensorflow as tf

from train_kfold import (
    MAX_LEN,
    Layout,
    add_handshape,
    add_rich_features,
    augment_batch,
    build_model,
    drop_z_indices,
    gate_hands,
    compute_device,
    load_sequences,
    pad_sequence,
    top_glosses,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "wlasl", "wlasl_gru.meta.json")
# La configuration exacte de ce script — pile complete ET opposition du pouce
# — n a jamais ete mesuree en k-fold. On cite donc la mesure la PLUS PROCHE,
# la pile complete sans l opposition, et on dit dans la fiche ce qui manque.
# Citer wlasl_kfold_shape8 (0.5998) serait pire : c est l opposition SANS la
# pile, donc une sous-estimation d environ 8 points.
KFOLD_PATH = os.path.join(MODELS_DIR, "wlasl", "wlasl_kfold_best_4folds.json")
ASLC_KFOLD_PATH = os.path.join(MODELS_DIR, "aslc_kfold_baseline.json")
KFOLD_CAVEAT = (
    "mesure sur 4 des 5 folds, au MEME pre-entrainement que ce modele "
    "(300 gloses, 120 epochs) mais a 3 graines au lieu de 5. Le fold 5 a ete "
    "tue par un redemarrage Windows et n a pas ete repris ; c etait le fold "
    "le plus facile du run de reference (0.794 contre une moyenne de 0.728 "
    "sur les quatre autres), donc le chiffre a 5 folds serait plutot "
    "au-dessus. Expedier a 5 graines plutot que 3 va aussi dans ce sens : "
    "l ensemble reduit la variance, donc le vrai modele est un peu meilleur "
    "que ce qui est annonce ici. Gain mesure du pre-entrainement elargi : "
    "+0.0341, t=3.20, 3 folds sur 4."
)


def stored_facts(features_dir):
    """Ce que les clips disent de leur extraction, verifie sur tout le dossier.

    La demo applique ce qui est ecrit dans la fiche : base de temps, duree
    d enregistrement, detecteur. Ces faits sont LUS dans les clips, pas
    supposes, et un dossier qui melange deux reponses est refuse — un modele
    entraine a moitie sur CUDA et a moitie sur TensorRT n aurait pas de bon
    reglage de demo.
    """
    detectors, timed = set(), set()
    for path in glob.glob(os.path.join(features_dir, "*.npz")):
        data = np.load(path, allow_pickle=True)
        detectors.add(str(data["detector"]) if "detector" in data.files else "cuda")
        timed.add("stamps" in data.files)
    if len(detectors) > 1 or len(timed) > 1:
        raise SystemExit(f"{features_dir} melange des extractions : "
                         f"detecteurs {sorted(detectors)}, horodatages {sorted(timed)}")
    centred = timed.pop() if timed else False
    return {
        "detector": detectors.pop() if detectors else "cuda",
        "timebase": centred,
        # 4 s pour laisser a l utilisateur le temps de commencer : la fenetre
        # de 80 frames est ensuite centree sur son signe, ou qu il tombe.
        "record_seconds": 4.0 if centred else 2.8,
    }


def extraction_threshold(features_dir):
    """Le seuil avec lequel le dossier a ete extrait, lu dans les npz.

    Il n est pas egal a --gate : en mode brut l extraction n ecarte rien,
    mais elle calcule quand meme le drapeau de conflit, et elle le fait a CE
    seuil. La demo doit rejouer le conflit a la meme valeur, faute de quoi
    elle marquerait des conflits sur d autres frames qu a l entrainement.
    """
    values = set()
    for path in sorted(glob.glob(os.path.join(features_dir, "*.npz")))[:50]:
        data = np.load(path, allow_pickle=True)
        if "hand_threshold" in data.files:
            values.add(round(float(data["hand_threshold"]), 4))
    if len(values) > 1:
        raise SystemExit(
            f"{features_dir} melange plusieurs seuils d extraction : "
            f"{sorted(values)}. Reextraire avant d entrainer.")
    return values.pop() if values else None


def dataset_of(features_dir):
    """'aslc' pour ASL Citizen, 'wlasl' sinon : le prefixe des fichiers."""
    return "aslc" if "asl_citizen" in os.path.normpath(features_dir) else "wlasl"


def model_path(seed_index, extractor="rtmpose", snapshot=None, dataset="wlasl"):
    """Un nom par jeu de donnees, par extracteur, par graine et par instantane.

    Les configurations ecrivaient au MEME chemin, et comme *.keras est ignore
    par git, entrainer l une ecrasait definitivement les poids de l autre.
    C est arrive une fois entre extracteurs ; sans le prefixe de jeu de
    donnees, le premier entrainement sur ASL Citizen aurait efface les 25
    modeles WLASL que la demo sert.
    """
    suffix = "" if extractor == "mediapipe" else f"_{extractor}"
    mark = "" if snapshot is None else f"_k{snapshot}"
    return os.path.join(dataset_dir(dataset),
                        f"{dataset}_gru{suffix}_s{seed_index}{mark}.keras")


def meta_path(extractor="rtmpose", dataset="wlasl"):
    suffix = "" if extractor == "mediapipe" else f"_{extractor}"
    return os.path.join(dataset_dir(dataset), f"{dataset}_gru{suffix}.meta.json")


def dataset_dir(dataset):
    """ASL Citizen, la phase en cours, a la racine de models/ ; WLASL, close,
    dans models/wlasl/ avec ses resultats."""
    return MODELS_DIR if dataset == "aslc" else os.path.join(MODELS_DIR, "wlasl")


def snapshot_epochs(total, count):
    """Les epochs ou figer les poids, reparties sur le plateau.

    PAR POSITION, jamais par score : choisir les "meilleures" epochs serait
    une selection sur le fold de test, et ici il n y a aucun ensemble de
    validation — le seul signal classable serait la precision
    d entrainement, qui favoriserait les instantanes les plus surappris.

    Le +2.3 mesure venait de la moyenne des predictions sur les 30
    DERNIERES epochs (section 7.8), donc on repartit sur cette fenetre.
    """
    if count <= 1:
        return [total]
    window = min(30, total)
    step = (window - 1) / (count - 1)
    return sorted({total - round(i * step) for i in range(count)})


def train_one(seed, features, labels, num_classes, args, pretrain=None,
              keep_columns=None, layout=None, on_snapshot=None):
    """features arrive en colonnes BRUTES (104 + queue), deja filtrees.

    Le retrait de la profondeur se fait JUSTE avant le modele : mirror(),
    appele par augment_batch(), code en dur les indices des deux mains, des
    drapeaux et de la pose sur la disposition complete. Couper d'abord
    decale tout et leve une erreur de forme — exactement l'erreur deja
    corrigee dans train_kfold.py, puis reintroduite ici.

    on_snapshot(index, model) est appele aux epochs rendues par
    snapshot_epochs ; c est l appelant qui decide quoi en faire.
    """
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    width = 26 if args.handshape_full else 8

    def to_model(batch):
        # Les features derivees se calculent sur la disposition COMPLETE
        # (indices des mains et de la pose codes en dur), donc avant la
        # selection de colonnes.
        # MEME ORDRE que train_kfold.to_model : d abord les 23 scalaires
        # derives, ensuite la configuration de main. Un ordre different
        # donnerait au modele les bonnes valeurs dans les mauvaises colonnes,
        # sans erreur de forme pour le signaler.
        pieces = []
        if args.rich:
            pieces.append(add_rich_features(batch, layout)[:, :, -23:])
        if args.handshape or args.handshape_full:
            pieces.append(add_handshape(batch, layout,
                                        full=args.handshape_full)[:, :, -width:])

        if keep_columns is not None:
            batch = batch[:, :, keep_columns]

        if pieces:
            batch = np.concatenate([batch] + pieces, axis=-1)

        return batch

    feature_dim = ((len(keep_columns) if keep_columns is not None
                    else features.shape[2])
                   + (23 if args.rich else 0)
                   + (width if (args.handshape or args.handshape_full) else 0))

    # raw_dim : les colonnes brutes, que le ST-GCN lit comme des articulations.
    raw_dim = len(keep_columns) if keep_columns is not None else features.shape[2]
    model = build_model(num_classes, args.units, feature_dim,
                        args.cell, args.label_smoothing, args.project,
                        raw_dim=raw_dim, readout=args.readout,
                        layer_norm=args.layer_norm)

    if pretrain is not None:
        pre_x, pre_y, pre_classes = pretrain
        head = build_model(pre_classes, args.units, feature_dim,
                           args.cell, args.label_smoothing, args.project,
                           raw_dim=raw_dim, readout=args.readout,
                           layer_norm=args.layer_norm)

        for index in range(args.pretrain_epochs):
            order = rng.permutation(len(pre_x))
            batch = to_model(augment_batch(pre_x[order], rng, layout))
            target = pre_y[order]
            if args.label_smoothing > 0:
                target = tf.keras.utils.to_categorical(target, pre_classes)
            history = head.fit(batch, target, batch_size=args.batch,
                               epochs=1, verbose=0)
            accuracy = history.history["accuracy"][0]
            if (index + 1) % 10 == 0:
                print(f"    pre-entrainement {index + 1}/"
                      f"{args.pretrain_epochs}  acc {accuracy:.3f}", flush=True)
            # Le fold 5 de la baseline ASL Citizen est reste au hasard pendant
            # tout son pre-entrainement (0.004 a e60, pour 0.72-0.76 ailleurs),
            # et l affinage qui en partait predisait une seule classe. Un
            # modele livre dans cet etat ne reconnaitrait rien : on s arrete
            # des que c est visible plutot qu apres une heure de calcul.
            if index + 1 == 10 and accuracy < 5.0 / pre_classes:
                raise SystemExit(
                    f"pre-entrainement effondre (acc {accuracy:.4f} a 10 "
                    f"epochs, hasard {1 / pre_classes:.4f}) : le GRU s est "
                    f"sature, voir build_model. Relancer avec une autre --seed, "
                    f"ou --layer-norm.")

        # Tout sauf la couche de sortie, dont le nombre de classes differe.
        for target_layer, source_layer in zip(model.layers[:-1], head.layers[:-1]):
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())

    marks = snapshot_epochs(args.epochs, args.snapshots)

    for epoch in range(args.epochs):
        order = rng.permutation(len(features))
        batch = to_model(augment_batch(features[order], rng, layout))
        target = labels[order]
        if args.label_smoothing > 0:
            target = tf.keras.utils.to_categorical(target, num_classes)

        history = model.fit(batch, target, batch_size=args.batch,
                            epochs=1, verbose=0)

        if (epoch + 1) in marks and on_snapshot is not None:
            on_snapshot(marks.index(epoch + 1), model, epoch + 1)

        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch + 1}/{args.epochs}  "
                  f"train_acc {history.history['accuracy'][0]:.3f}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=96)
    parser.add_argument("--deterministic", action="store_true",
                        help="calcul reproductible au bit pres (GPU) ; sans "
                             "lui, --only-fold ne reproduit plus un fold "
                             "interrompu sur GPU")
    parser.add_argument("--cell", default="gru",
                        choices=("gru", "lstm", "stack", "tcn", "stgcn"),
                        help="architecture ; voir build_model dans train_kfold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=100,
                        help="les N premiers signes par rang ASL-LEX (ASL "
                             "Citizen), ou les N plus fournis (WLASL)")
    parser.add_argument("--variants", default="all",
                        help="WLASL seulement : ASL Citizen numerote deja ses "
                             "variantes (CANDY1, CANDY2), chacune sa classe")
    parser.add_argument("--features-dir",
                        default=os.path.join(PROJECT_ROOT, "data", "asl_citizen",
                                             "features"),
                        help="dossier de features ; par defaut ASL Citizen")
    parser.add_argument("--vocab-from", default=None,
                        help="dossier dont on reprend les N gloses cibles "
                             "(WLASL : data/wlasl/features). Inutile pour ASL "
                             "Citizen, ou le rang est dans chaque clip.")
    parser.add_argument("--kfold", default=None,
                        help="resultat k-fold a citer dans la fiche. Refuse "
                             "s il a ete mesure sur un autre dossier de "
                             "features. Sans lui, la fiche dit que la "
                             "precision n est pas encore mesuree.")
    parser.add_argument("--handshape", action="store_true", default=True,
                        help="8 scalaires d opposition du pouce")
    parser.add_argument("--handshape-full", action="store_true", default=True,
                        help="26 scalaires de configuration de main "
                             "(remplace les 8)")
    parser.add_argument("--confidence", action="store_true", default=True,
                        help="confiance de chaque main en valeur continue")
    parser.add_argument("--no-z", action="store_true", default=False,
                        help="MediaPipe seulement : RTMPose n a pas de z")
    parser.add_argument("--rich", action="store_true", default=True,
                        help="23 scalaires derives de x,y")
    parser.add_argument("--project", type=int, default=48,
                        help="projection lineaire avant le GRU : 48 params "
                             "par dimension au lieu de 576 (+4.7 points)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--snapshots", type=int, default=5,
                        help="poids figes par graine, repartis sur les 30 "
                             "dernieres epochs. Gratuits a l entrainement.")
    parser.add_argument("--gate", type=float, default=0.7,
                        help="confiance minimale d une main. features_rtm_all "
                             "etant extrait en mode brut, c est ICI que le "
                             "filtrage a lieu, pas a l extraction.")
    parser.add_argument("--gate-inside", type=float, default=0.5,
                        help="part minimale des points de main dans l image")
    parser.add_argument("--gate-conflict", action="store_true", default=True,
                        help="ecarter la main inventee quand les deux jeux de "
                             "points tombent sur la meme main")
    parser.add_argument("--gate-soft", action="store_true", default=False,
                        help="garder les coordonnees, ne baisser que le "
                             "drapeau (mesure moins bon que le filtrage dur)")
    parser.add_argument("--pretrain-vocab", type=int, default=400,
                        help="les N premiers signes, cible comprise : rangs "
                             "1 a 400 pour ASL Citizen")
    parser.add_argument("--pretrain-epochs", type=int, default=60,
                        help="60 pour la baseline ASL Citizen ; le modele WLASL "
                             "livre en avait 120")
    parser.add_argument("--gate-pretrain", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="filtrer l echafaudage comme la cible. Oui pour la "
                             "baseline ASL Citizen : non filtrees, 6%% de ses "
                             "frames portent des mains repliees aux coordonnees "
                             "absurdes. Le modele WLASL livre etait entraine "
                             "sans (--no-gate-pretrain).")
    parser.add_argument("--readout", default="attention",
                        choices=("last", "mean", "attention"),
                        help="comment le GRU rend sa reponse. 'attention' pour "
                             "ASL Citizen : l etat final y arrive apres ~18 "
                             "frames de repos et ne sort pas du hasard. Le "
                             "modele WLASL livre utilisait 'last'.")
    parser.add_argument("--layer-norm", action="store_true",
                        help="LayerNormalization avant le GRU (voir "
                             "build_model) ; pas dans la baseline")
    parser.add_argument("--pretrain-variants", default="all",
                        choices=("", "all", "dominant", "split"),
                        help="filtrage des variantes de l echafaudage seul. "
                             "'all' lui rend 396 clips que 'dominant' ecarte, "
                             "au prix d etiquettes contradictoires dont "
                             "l echafaudage n a pas besoin.")
    args = parser.parse_args()
    device = compute_device(args.deterministic)
    print(f"calcul sur {device}")

    gating = bool(args.gate or args.gate_inside or args.gate_conflict)

    records = load_sequences(args.variants, args.features_dir,
                             with_confidence=args.confidence,
                             with_meta=gating)

    # Les 50 cibles viennent du dossier de reference, pas des 300 gloses
    # presentes ici : sinon on entrainerait sur un autre vocabulaire que
    # celui mesure, sans que rien ne le signale.
    source = args.vocab_from or args.features_dir
    # Par rang ASL-LEX si les clips en portent un, sinon par nombre de clips.
    target_glosses = top_glosses(
        load_sequences(args.variants, source, metadata_only=True), args.vocab)

    glosses = sorted(target_glosses)
    index = {g: i for i, g in enumerate(glosses)}

    selected = [r for r in records if r["gloss"] in target_glosses]
    absent = sorted(target_glosses - {r["gloss"] for r in selected})
    if absent:
        raise SystemExit(
            f"{len(absent)} gloses cibles absentes de {args.features_dir} : "
            f"{absent}")

    features = np.stack([pad_sequence(r["features"]) for r in selected])
    labels = np.array([index[r["gloss"]] for r in selected], dtype=np.int32)

    keep_columns = drop_z_indices(features.shape[2]) if args.no_z else None

    # La disposition ne peut pas etre devinee quand il y a une queue : 106
    # colonnes se lisent aussi bien comme 9 points de pose plus 2 confiances
    # que comme 10 points de pose, et le second ferait negater une confiance
    # au miroir. Voir Layout.from_dim.
    tail = 2 if args.confidence else 0
    layout = Layout.from_dim(features.shape[2] - tail)
    layout.tail = tail

    width = 26 if args.handshape_full else 8
    feature_dim = ((len(keep_columns) if keep_columns is not None
                    else features.shape[2])
                   + (23 if args.rich else 0)
                   + (width if (args.handshape or args.handshape_full) else 0))

    def apply_gate(batch, items, label):
        """Le filtrage, a l identique du k-fold.

        En mode brut les deux mains sont toujours ecrites, drapeaux a 1 :
        sans cet appel le modele verrait des mains que la mesure lui avait
        cachees, et la demo filtrerait une entree que l entrainement n avait
        jamais filtree.
        """
        scores = np.stack([pad_sequence(r["meta"]["hand_scores"])
                           for r in items])
        inside = np.stack([pad_sequence(r["meta"]["hand_inside"])
                           for r in items])
        conflict = np.stack([
            pad_sequence(
                r["meta"]["hand_conflict"].astype(np.float32)[:, None])[:, 0]
            for r in items]).astype(bool)

        def kept(array):
            return float((array[..., layout.left_flag] > 0.5).mean()
                         + (array[..., layout.right_flag] > 0.5).mean()) / 2.0

        before = kept(batch)
        out = gate_hands(batch, layout, scores, inside, conflict,
                         threshold=args.gate, soft=args.gate_soft,
                         inside_threshold=args.gate_inside,
                         use_conflict=args.gate_conflict)
        print(f"filtrage {label} : seuil {args.gate}"
              f"{' doux' if args.gate_soft else ' dur'}"
              f"{f', hors-cadre {args.gate_inside}' if args.gate_inside else ''}"
              f"{', conflits' if args.gate_conflict else ''}"
              f"   mains gardees {before:.1%} -> {kept(out):.1%}")
        return out

    if gating:
        features = apply_gate(features, selected, "cible")

    own = collections.Counter(r["gloss"] for r in selected)
    print(f"{len(selected)} sequences, {len(glosses)} gloses, "
          f"{feature_dim} dimensions, projection {args.project}")
    print(f"{args.seeds} graines x {args.snapshots} instantanes = "
          f"{args.seeds * args.snapshots} modeles, {args.epochs} epochs")
    print()

    pretrain = None
    if args.pretrain_vocab:
        # Chargement SEPARE : l echafaudage a son propre filtrage de
        # variantes, et il couvre les 300 gloses du dossier, pas les 50
        # cibles. Meme dossier, donc memes colonnes et meme extracteur.
        chosen = args.pretrain_variants or args.variants
        extra = load_sequences(chosen, args.features_dir,
                               with_confidence=args.confidence,
                               with_meta=gating)
        wide = top_glosses(extra, args.pretrain_vocab)
        extra = [r for r in extra if r["gloss"] in wide]
        wide_index = {g: i for i, g in enumerate(sorted(wide))}

        pre_x = np.stack([pad_sequence(r["features"]) for r in extra])
        if gating and args.gate_pretrain:
            # Hors drapeau, l echafaudage voit les features BRUTES. C etait
            # tentant de le filtrer comme la cible, et je l avais fait : mais
            # train_kfold ne l a jamais fait, donc le faire ici rendait ce
            # modele different de celui qu on mesure. Visible dans les
            # courbes : e120 0.745 filtre contre 0.527 non filtre.
            pre_x = apply_gate(pre_x, extra, "echafaudage")

        pretrain = (
            pre_x,
            np.array([wide_index[r["gloss"]] for r in extra], dtype=np.int32),
            len(wide),
        )
        print(f"pre-entrainement : {len(extra)} sequences, {len(wide)} gloses, "
              f"variantes {chosen}, {args.pretrain_epochs} epochs")
        print()

    # L extracteur se deduisait du nom du dossier ("rtm" dedans) :
    # data/asl_citizen/features n en contient pas, et le modele aurait ete
    # etiquete MediaPipe, que la demo RTMPose refuse. Les clips qui portent un
    # detecteur viennent forcement de RTMPose.
    dataset = dataset_of(args.features_dir)
    extractor = ("rtmpose" if "rtm" in args.features_dir or dataset == "aslc"
                 else "mediapipe")

    # La fiche cite une mesure, et seulement une mesure faite sur les memes
    # features ET avec la meme recette. Verifie AVANT d entrainer : une
    # recette qui differe se decouvrait sinon apres une heure de calcul.
    kfold = None
    kfold_path = args.kfold
    if kfold_path is None:
        kfold_path = ASLC_KFOLD_PATH if dataset == "aslc" else KFOLD_PATH
    if kfold_path and os.path.exists(kfold_path):
        with open(kfold_path, encoding="utf-8") as handle:
            kfold = json.load(handle)
        measured_on = kfold.get("features_dir")
        here = os.path.relpath(args.features_dir, PROJECT_ROOT)
        if measured_on is not None and os.path.normpath(measured_on) != os.path.normpath(here):
            raise SystemExit(
                f"{kfold_path} a ete mesure sur {measured_on}, ce modele est "
                f"entraine sur {here} : la fiche citerait la precision d un "
                f"autre modele.")
        mine = {"cell": args.cell, "epochs": args.epochs, "units": args.units,
                "project": args.project, "rich": bool(args.rich),
                "confidence": bool(args.confidence),
                "handshape_full": bool(args.handshape_full),
                "label_smoothing": args.label_smoothing, "vocab": args.vocab,
                "pretrain_vocab": args.pretrain_vocab,
                "pretrain_epochs": args.pretrain_epochs,
                "gate_pretrain": bool(args.gate_pretrain), "gate": args.gate,
                "gate_inside": args.gate_inside,
                "gate_conflict": bool(args.gate_conflict),
                "gate_soft": bool(args.gate_soft), "readout": args.readout,
                "layer_norm": bool(args.layer_norm)}
        # Seulement ce que le k-fold enregistre : les anciens resultats WLASL
        # ne portent pas ces cles.
        differs = {k: (kfold[k], v) for k, v in mine.items()
                   if k in kfold and kfold[k] != v}
        if differs:
            raise SystemExit(
                f"{kfold_path} mesure une autre recette que celle-ci : "
                + ", ".join(f"{k} {a!r} la-bas, {b!r} ici"
                            for k, (a, b) in differs.items())
                + ". Passer les memes options, ou --kfold vers la bonne mesure.")

    saved = []
    for seed_index in range(args.seeds):
        print(f"  graine {seed_index + 1}/{args.seeds}")

        def keep_snapshot(snapshot_index, model, epoch, _seed=seed_index):
            path = model_path(_seed, extractor, snapshot_index, dataset)
            model.save(path)
            saved.append((os.path.relpath(path, PROJECT_ROOT), epoch))

        train_one(args.seed + seed_index * 100, features, labels,
                  len(glosses), args, pretrain, keep_columns, layout,
                  on_snapshot=keep_snapshot)

    # Avant le garde-fou du dossier, la fiche citait TOUJOURS le k-fold
    # WLASL : un modele ASL Citizen serait parti en annoncant 0.7623, chiffre
    # mesure sur un autre jeu de donnees.
    expected = None
    if kfold is not None:
        # per_fold_plateau, jamais per_fold : le second est la derniere epoch,
        # et confondre les deux a deja produit un resultat de signe faux
        # (section 7.6).
        plateau = np.array(kfold["per_fold_plateau"])
        expected = {
            "mean": round(float(plateau.mean()), 4),
            "std": round(float(plateau.std()), 4),
            "folds": kfold["folds"],
            "chance": round(1 / len(glosses), 4),
            "measured_config": os.path.basename(kfold_path),
            "caveat": KFOLD_CAVEAT if kfold_path == KFOLD_PATH else None,
        }

    metadata = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "gru": "ensemble de GRU bidirectionnels 2 couches",
            "lstm": "ensemble de LSTM bidirectionnels 2 couches",
            "stack": "ensemble GRU puis LSTM bidirectionnels",
            "tcn": "ensemble de TCN a convolutions dilatees non causales",
            "stgcn": "ensemble de ST-GCN sur les 51 articulations du squelette",
        }[args.cell] + " sur sequences de landmarks",
        "cell": args.cell,
        "readout": args.readout,
        "layer_norm": bool(args.layer_norm),
        "device": device,
        "models": [path for path, _ in saved],
        "snapshot_epochs": [epoch for _, epoch in saved],
        "classes": glosses,
        "num_classes": len(glosses),
        "sequences": len(selected),
        "clips_per_gloss": {g: own[g] for g in glosses},
        "max_len": MAX_LEN,
        "feature_dim": int(feature_dim),
        "drop_z": bool(args.no_z),
        "rich": bool(args.rich),
        "confidence": bool(args.confidence),
        "handshape": bool(args.handshape or args.handshape_full),
        "handshape_full": bool(args.handshape_full),
        "project": int(args.project),
        # La demo DOIT rejouer ce filtrage : les features d entrainement sont
        # brutes, donc ces quatre valeurs font partie de la definition de
        # l entree du modele, et non d un reglage de l extraction.
        "raw_features": True,
        "extraction_threshold": extraction_threshold(args.features_dir),
        **stored_facts(args.features_dir),
        "gate": float(args.gate),
        "gate_inside": float(args.gate_inside),
        "gate_conflict": bool(args.gate_conflict),
        "gate_soft": bool(args.gate_soft),
        "features_dir": os.path.relpath(args.features_dir, PROJECT_ROOT),
        "vocab_from": (os.path.relpath(args.vocab_from, PROJECT_ROOT)
                       if args.vocab_from else None),
        "extractor": extractor,
        "keep_columns": keep_columns.tolist() if keep_columns is not None else None,
        "seeds": args.seeds,
        "snapshots": args.snapshots,
        "pretrain_epochs": args.pretrain_epochs,
        "units": args.units,
        "epochs": args.epochs,
        "label_smoothing": args.label_smoothing,
        "variants": args.variants,
        "pretrain_vocab": args.pretrain_vocab,
        "pretrain_variants": args.pretrain_variants or args.variants,
        "gate_pretrain": bool(args.gate_pretrain),
        "expected_accuracy": expected,
        "accuracy_note": (
            "expected_accuracy vient de la validation croisee par signeur "
            "(train_kfold.py, configuration identique), pas de ce modele : "
            "celui-ci est entraine sur toutes les donnees et n'a pas "
            "d'ensemble de test propre."
        ),
    }

    with open(meta_path(extractor, dataset), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print()
    print(f"  {len(saved)} modeles sauvegardes")
    print(f"  fiche : {meta_path(extractor, dataset)}")
    if expected:
        print(f"  precision attendue (k-fold) : {expected['mean']:.4f} "
              f"+- {expected['std']:.4f}  (hasard {expected['chance']})")


if __name__ == "__main__":
    main()
