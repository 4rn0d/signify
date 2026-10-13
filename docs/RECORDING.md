# Enregistrement des 50 signes — consignes pour les signeurs

Merci de participer. Ces enregistrements servent a apprendre a un modele
a reconnaitre 50 signes ASL a partir d une webcam.

## La chose la plus importante

**Signez la variante montree dans le clip de reference, pas celle que vous
connaissez.** Plusieurs de ces mots ont plusieurs facons d etre signes, et
le modele n a appris qu une seule variante de chacun. Une autre variante
compte pour lui comme un autre mot, et l enregistrement serait inutilisable.

Regardez le clip avant chaque serie. Les clips sont dans `data/wlasl/reference/`,
deux par mot (`<mot>_1.mp4` et `<mot>_2.mp4`), par deux personnes differentes :
ce qu elles ont en commun est ce qui compte.

**9 mots sur 50 ont plusieurs variantes dans les donnees** et
demandent une attention particuliere :

| mot | variante utilisee |
| --- | --- |
| before | 0 sur 2 (12/22 clips) |
| candy | 1 sur 2 (11/18 clips) |
| corn | 0 sur 2 (11/13 clips) |
| dark | 0 sur 2 (11/14 clips) |
| fine | 0 sur 2 (12/15 clips) |
| finish | 0 sur 2 (11/16 clips) |
| go | 0 sur 2 (14/18 clips) |
| like | 0 sur 2 (14/18 clips) |
| shirt | 0 sur 2 (12/16 clips) |

## Protocole

- **Cadrage** : buste visible, des hanches au-dessus de la tete. Les deux mains
  doivent rester dans l image pendant tout le signe.
- **Prises par mot** : 3 normalement, **6 pour les mots marques (!)**.
- **Variez d une prise a l autre** :

  | prise | distance | angle | vitesse |
  | --- | --- | --- | --- |
  | 1 | normale | face | normale |
  | 2 | proche | face | normale |
  | 3 | loin | 3/4 | normale |
  | 4 | normale | face | rapide |
  | 5 | proche | 3/4 | normale |
  | 6 | loin | face | rapide |

  Les prises 4 a 6 ne concernent que les mots marques. La distance compte le
  plus : c est la taille des mains en pixels qui limite aujourd hui la capacite
  du modele a distinguer les configurations de doigts. Inutile de changer de
  piece ou d eclairage : le modele ne voit que des coordonnees d articulations,
  pas l image.

- **Entre deux prises**, revenez les bras au repos le long du corps, puis
  marquez une pause d une seconde. Le decoupage en depend.
- **Notez votre identifiant de signeur** sur tous vos fichiers : c est ce qui
  permet de tester le modele sur une personne qu il n a jamais vue.

## Les 50 mots

`(!)` = mot facilement confondu avec un autre, a signer bien nettement, et
6 prises au lieu de 3.

| mot | variante | reference | prises | confondu avec |
| --- | --- | --- | --- | --- |
| **apple** (!) | 0 | apple_1.mp4, apple_2.mp4 | 6 | candy |
| **basketball** | 0 | basketball_1.mp4, basketball_2.mp4 | 3 |  |
| **bed** (!) | 0 | bed_1.mp4, bed_2.mp4 | 6 | mother |
| **before** | 0 | before_1.mp4, before_2.mp4 | 3 |  |
| **bird** (!) | 0 | bird_1.mp4, bird_2.mp4 | 6 | orange |
| **black** | 0 | black_1.mp4, black_2.mp4 | 3 |  |
| **blue** | 0 | blue_1.mp4, blue_2.mp4 | 3 |  |
| **book** | 0 | book_1.mp4, book_2.mp4 | 3 |  |
| **bowling** | 0 | bowling_1.mp4, bowling_2.mp4 | 3 |  |
| **can** | 0 | can_1.mp4, can_2.mp4 | 3 |  |
| **candy** (!) | 1 | candy_1.mp4, candy_2.mp4 | 6 | apple |
| **chair** | 0 | chair_1.mp4, chair_2.mp4 | 3 |  |
| **change** (!) | 0 | change_1.mp4, change_2.mp4 | 6 | year |
| **color** | 0 | color_1.mp4, color_2.mp4 | 3 |  |
| **corn** | 0 | corn_1.mp4, corn_2.mp4 | 3 |  |
| **cousin** (!) | 0 | cousin_1.mp4, cousin_2.mp4 | 6 | hat |
| **dance** (!) | 0 | dance_1.mp4, dance_2.mp4 | 6 | study |
| **dark** | 0 | dark_1.mp4, dark_2.mp4 | 3 |  |
| **deaf** (!) | 0 | deaf_1.mp4, deaf_2.mp4 | 6 | hearing |
| **drink** | 0 | drink_1.mp4, drink_2.mp4 | 3 |  |
| **eat** (!) | 0 | eat_1.mp4, eat_2.mp4 | 6 | wrong |
| **enjoy** | 0 | enjoy_1.mp4, enjoy_2.mp4 | 3 |  |
| **family** | 0 | family_1.mp4, family_2.mp4 | 3 |  |
| **fine** | 0 | fine_1.mp4, fine_2.mp4 | 3 |  |
| **finish** | 0 | finish_1.mp4, finish_2.mp4 | 3 |  |
| **forget** | 0 | forget_1.mp4, forget_2.mp4 | 3 |  |
| **go** | 0 | go_1.mp4, go_2.mp4 | 3 |  |
| **graduate** (!) | 0 | graduate_1.mp4, graduate_2.mp4 | 6 | help |
| **hat** (!) | 0 | hat_1.mp4, hat_2.mp4 | 6 | cousin |
| **hearing** (!) | 0 | hearing_1.mp4, hearing_2.mp4 | 6 | deaf |
| **help** (!) | 0 | help_1.mp4, help_2.mp4 | 6 | graduate |
| **hot** | 0 | hot_1.mp4, hot_2.mp4 | 3 |  |
| **like** (!) | 0 | like_1.mp4, like_2.mp4 | 6 | white |
| **man** | 0 | man_1.mp4, man_2.mp4 | 3 |  |
| **many** | 0 | many_1.mp4, many_2.mp4 | 3 |  |
| **mother** (!) | 0 | mother_1.mp4, mother_2.mp4 | 6 | bed |
| **no** | 0 | no_1.mp4, no_2.mp4 | 3 |  |
| **now** | 0 | now_1.mp4, now_2.mp4 | 3 |  |
| **orange** (!) | 0 | orange_1.mp4, orange_2.mp4 | 6 | bird |
| **play** | 0 | play_1.mp4, play_2.mp4 | 3 |  |
| **shirt** (!) | 0 | shirt_1.mp4, shirt_2.mp4 | 6 | thursday |
| **study** (!) | 0 | study_1.mp4, study_2.mp4 | 6 | dance |
| **thursday** (!) | 0 | thursday_1.mp4, thursday_2.mp4 | 6 | shirt |
| **walk** | 0 | walk_1.mp4, walk_2.mp4 | 3 |  |
| **white** (!) | 0 | white_1.mp4, white_2.mp4 | 6 | like |
| **who** | 0 | who_1.mp4, who_2.mp4 | 3 |  |
| **woman** | 0 | woman_1.mp4, woman_2.mp4 | 3 |  |
| **wrong** (!) | 0 | wrong_1.mp4, wrong_2.mp4 | 6 | eat |
| **year** (!) | 0 | year_1.mp4, year_2.mp4 | 6 | change |
| **yes** | 0 | yes_1.mp4, yes_2.mp4 | 3 |  |

**Total par signeur : 216 prises** (22 mots marques x 6 + 28 x 3).
A une quinzaine de secondes par prise, environ 54 minutes
de travail effectif, pauses comprises.

## Pourquoi ces mots-la demandent plus de prises

Le modele les confond aujourd hui parce qu ils se font au meme endroit du
corps avec un mouvement d ensemble semblable, et ne different que par la
configuration des doigts — par exemple `apple` et `candy` sont tous deux une
torsion a la joue. Sur les videos du dataset la main ne fait qu une quarantaine
de pixels, trop peu pour voir un index plie contre un index tendu. Sur une
webcam de pres, elle en fait environ 200. Ce sont donc precisement ces mots
que de nouveaux enregistrements peuvent reellement corriger.
