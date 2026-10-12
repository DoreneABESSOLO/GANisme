# GAN — Génération de peintures à partir du catalogue Web Gallery of Art

Projet de data science : construire, à partir du catalogue de la **Web Gallery of
Art** (WGA — art européen du 3ᵉ au 19ᵉ siècle), un jeu de données d'images
exploitable pour entraîner un **GAN (DCGAN)** capable de générer des peintures
inédites.

Le projet suit un pipeline en trois temps, chacun documenté par un notebook
Jupyter exécutable de bout en bout :

```
Catalogue WGA  →  EDA & sélection  →  Scraping des images  →  Prétraitement  →  Entraînement GAN
(52 867 œuvres)   (32 438 peintures)   (3 131 images)          (crop/normalisation)   (DCGAN, BCE)
```

Un support de présentation (`presentation/gan_peintures.pptx`) résume l'ensemble
avec les visualisations produites par les notebooks.

## Sommaire

- [Source des données](#source-des-données)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [1. Analyse exploratoire (EDA)](#1-analyse-exploratoire-eda)
  - [2. Récupération des images (scraping)](#2-récupération-des-images-scraping)
  - [3. Prétraitement](#3-prétraitement)
  - [4. Entraînement du GAN](#4-entraînement-du-gan)
- [Résultats actuels](#résultats-actuels)
- [Choix méthodologiques](#choix-méthodologiques)
- [Limites connues & prochaines étapes](#limites-connues--prochaines-étapes)
- [Considérations légales et éthiques](#considérations-légales-et-éthiques)

## Source des données

Le catalogue provient de la [Web Gallery of Art](https://www.wga.hu), une base
de données non commerciale maintenue par deux personnes, qui référence l'art
européen du 3ᵉ au 19ᵉ siècle (peinture, sculpture, architecture, arts
graphiques…). Il est fourni sous forme de fichier téléchargeable
(`data/art_catalog.xlsx` / `catalog.csv`) et contient **52 867 œuvres**
décrites par 11 champs texte : `AUTHOR`, `BORN-DIED`, `TITLE`, `DATE`,
`TECHNIQUE`, `LOCATION`, `URL`, `FORM`, `TYPE`, `SCHOOL`, `TIMEFRAME`.

Le catalogue ne contient **aucune image** : seules des URL vers des pages
HTML de présentation sur wga.hu sont fournies — la récupération des images
fait partie intégrante du projet (voir [étape 2](#2-récupération-des-images-scraping)).

## Structure du projet

```
GAN/
├── README.md                      # ce fichier
│
├── eda_peintures.ipynb            # notebook d'analyse exploratoire (EDA)
├── eda_peintures.py                # script équivalent, non interactif
├── report_data.json                # agrégats produits par l'EDA
│
├── scraping_images.ipynb           # scraping local (VS Code / Jupyter)
├── scraping_images_colab.ipynb     # variante autonome pour Google Colab
├── scrape_images.py                # moteur de téléchargement (module partagé)
├── build_subset.py                 # échantillonnage stratifié (module partagé)
│
├── preprocessing_images.ipynb      # pipeline de prétraitement + visualisations
│
├── gan_dataset.py                   # Dataset PyTorch partagé (filtrage, crop, normalisation)
├── gan_models.py                    # architecture DCGAN (Générateur / Discriminateur)
├── train_gan.ipynb                  # notebook d'entraînement du GAN
│
├── data/
│   ├── art_catalog.xlsx             # catalogue source (téléchargé depuis wga.hu)
│   ├── art_catalog.xlsx - catalog.csv
│   ├── paintings_only.csv           # sous-ensemble FORM == "painting" (32 438 lignes)
│   └── subset_sample.csv            # sous-échantillon stratifié (3 680 lignes visées)
│
├── images/                          # images téléchargées (miroir de l'arborescence wga.hu)
│   └── _failed.csv                  # journal des échecs de téléchargement
│
├── checkpoints/                     # poids sauvegardés du GAN (generator.pt, discriminator.pt)
│
└── presentation/
    └── gan_peintures.pptx           # support de présentation du projet
```

> `EDA.ipynb` (à la racine, en majuscules) est une ébauche antérieure non liée
> à ce pipeline — à ignorer, conservée telle quelle sans modification.

## Installation

Python ≥ 3.10 recommandé. Dépendances principales :

```bash
pip install pandas numpy matplotlib pillow requests
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # build CPU
pip install nbformat nbclient ipykernel      # pour exécuter les notebooks en ligne de commande
```

Aucune clé d'API n'est nécessaire. Les notebooks s'exécutent aussi bien en
local (VS Code, Jupyter) que sur Google Colab (voir `scraping_images_colab.ipynb`
et la section « Compatibilité Colab » de `train_gan.ipynb`).

## Utilisation

### 1. Analyse exploratoire (EDA)

```bash
python eda_peintures.py
# ou, en notebook : ouvrir et exécuter eda_peintures.ipynb
```

Produit `data/paintings_only.csv` (sous-ensemble peintures) et
`report_data.json` (agrégats pour la présentation). Couvre : qualité des
données, répartition par genre/école/période, artistes, techniques et
dimensions, fiabilité du champ `DATE`, lieux de conservation.

**Constat principal :** le catalogue est propre au niveau structurel (aucun
doublon, aucun champ vide) mais fortement déséquilibré — 42 % de peinture
religieuse, 42 % d'école italienne, aucune œuvre après 1900.

### 2. Récupération des images (scraping)

Le mapping entre une page de présentation et son image pleine résolution est
systématique :

```
https://www.wga.hu/html/{a}/{artiste}/{oeuvre}.html
    → https://www.wga.hu/art/{a}/{artiste}/{oeuvre}.jpg
```

Construire un sous-échantillon stratifié (quota égal par genre puis par
école, pour corriger le déséquilibre de l'EDA) :

```bash
python build_subset.py --size 4000
```

Puis télécharger les images (débit volontairement limité — voir
[considérations légales](#considérations-légales-et-éthiques)) :

```bash
python scrape_images.py --csv data/subset_sample.csv --out images --workers 4 --min-delay 0.4
```

Options utiles : `--limit N` (test sur un échantillon), `--workers`,
`--min-delay`, `--timeout`. Le script reprend automatiquement là où il s'est
arrêté (les fichiers déjà présents sont ignorés) et journalise les échecs
dans `images/_failed.csv`.

Pour exécuter le scraping depuis Google Colab plutôt qu'en local, utiliser
`scraping_images_colab.ipynb` (autonome : il réécrit ses propres dépendances
via `%%writefile`, monte Google Drive pour persister les images entre
sessions).

### 3. Prétraitement

`preprocessing_images.ipynb` documente et visualise chaque étape sur les
images réellement téléchargées :

1. Validation des fichiers (détection des téléchargements corrompus)
2. Analyse des dimensions et ratios réels
3. Filtrage des formats extrêmes (ratio largeur/hauteur hors de 0,4–2,5, ou
   plus petit côté < 64 px)
4. Comparaison center-crop vs letterbox (center-crop retenu)
5. Normalisation (`RGB → crop/resize 64×64 ou 128×128 → tensor → [-1, 1]`)
6. `Dataset` / `DataLoader` PyTorch robuste (ignore une image illisible plutôt
   que de planter)
7. Augmentation optionnelle (flip horizontal, avec réserve sur les
   conventions picturales)

La logique réutilisable (dataset, filtrage, center-crop) est factorisée dans
`gan_dataset.py`, importé à la fois par ce notebook et par `train_gan.ipynb`.

### 4. Entraînement du GAN

```bash
# ouvrir et exécuter train_gan.ipynb (local ou Colab)
```

Points clés du notebook :

- **Données** : utilise en priorité `data/subset_sample.csv` (sous-échantillon
  stratifié) plutôt qu'un mélange arbitraire de tout `images/`, via
  `gan_dataset.subset_image_paths()`.
- **Reprise automatique** : si `checkpoints/generator.pt` existe déjà, les
  poids et l'historique des loss sont rechargés, et l'entraînement reprend à
  l'epoch suivante — utile pour un entraînement long en CPU, réparti sur
  plusieurs sessions.
- **Checkpoints périodiques** (`CKPT_EVERY`, 25 epochs par défaut) :
  limite la perte de progression en cas d'interruption.
- **Visualisation de la progression** : un instantané généré à partir d'un
  bruit latent fixe est affiché régulièrement, pour suivre visuellement la
  convergence du générateur.

En CPU, un entraînement complet (plusieurs centaines d'epochs sur quelques
milliers d'images) peut prendre plusieurs heures ; `train_gan.ipynb` contient
une section « Compatibilité Colab » pour migrer l'exécution — pas le
scraping — sur le GPU gratuit de Colab si besoin.

## Résultats actuels

| | |
|---|---|
| Œuvres dans le catalogue | 52 867 |
| Peintures sélectionnées (`FORM == "painting"`) | 32 438 (61,4 %) |
| Sous-échantillon stratifié visé | 3 680 images |
| Images effectivement téléchargées | 2 973 réussies / 3 680 (81 %), 3 131 au total disponibles |
| Checkpoints du GAN sauvegardés | `checkpoints/generator.pt`, `checkpoints/discriminator.pt` |

Un premier run de validation (DCGAN, 30 epochs, 154 images, CPU local) a
confirmé que le pipeline entraîne correctement : la loss du discriminateur
reste stable (jamais écrasée à zéro), celle du générateur oscille sans
exploser, et les générations évoluent visiblement du bruit pur vers des
textures et couleurs cohérentes — sans toutefois produire de formes
reconnaissables, ce qui est attendu à cette échelle de données/epochs. Des
runs complémentaires ont depuis été exécutés localement sur un jeu d'images
plus large ; les résultats détaillés de ces runs ne sont pas encore
consolidés dans ce dépôt.

Le détail (courbes de loss, grilles de progression, comparaison réel/généré)
est visualisé dans `train_gan.ipynb` et repris dans
`presentation/gan_peintures.pptx`.

## Choix méthodologiques

- **Architecture : DCGAN** plutôt que StyleGAN ou un modèle de diffusion —
  seule architecture réaliste à entraîner en CPU-only sur un corpus de
  quelques milliers d'images ; bien documentée, point de départ standard.
- **Fonction de perte : BCE non-saturante** (le générateur maximise
  `log D(G(z))` plutôt que de minimiser `log(1-D(G(z)))`, pour éviter des
  gradients quasi nuls en début d'entraînement) + **label smoothing** (0,9
  côté réel) pour limiter la surconfiance du discriminateur, facteur connu de
  mode collapse. Hyperparamètres Adam (`lr=2e-4`, `β=(0.5, 0.999)`)
  recommandés par le papier DCGAN original.
- **Center-crop plutôt que letterbox** pour la mise en forme des images :
  les formats extrêmes étant déjà filtrés en amont, le center-crop perd peu
  de contenu réel et évite d'introduire des bordures artificielles que le
  modèle pourrait apprendre à tort.
- **Échantillonnage stratifié** (quota égal par genre puis par école) pour
  atténuer le déséquilibre structurel du catalogue plutôt que de l'ignorer.

## Limites connues & prochaines étapes

- **Échelle actuelle limitée** : les résultats documentés viennent d'un run
  de validation (154 images, 30 epochs) qui prouve la mécanique du pipeline
  mais ne produit pas encore de peintures convaincantes. Un entraînement
  complet nécessite plusieurs milliers d'images et plusieurs centaines
  d'epochs.
- **Déséquilibre du corpus** : même stratifié, l'échantillon reste dominé par
  l'école italienne et le genre religieux. Un GAN conditionnel (sur `TYPE`
  et/ou `SCHOOL`) est une piste pour mieux contrôler la diversité générée.
- **~18 % d'échecs de téléchargement** (timeouts) sur le scraping complet du
  sous-échantillon — dus à la capacité limitée du serveur WGA, pas à un
  blocage ; relancer `scrape_images.py` avec le même `--csv` complète
  automatiquement les manquants.
- **Fiabilité du champ `DATE`** : texte libre très hétérogène (seulement 32 %
  d'années exactes) — préférer systématiquement `TIMEFRAME` pour tout
  filtrage ou conditionnement temporel.
- **Mode collapse à surveiller** à plus grande échelle : si les courbes de
  loss dérivent, basculer vers une **hinge loss** ou une **LSGAN** est un
  changement localisé (une cellule de `train_gan.ipynb`), pas une
  réécriture.

## Considérations légales et éthiques

La Web Gallery of Art précise (`wga.hu/legal.html`) que les images
téléchargées depuis le site sont destinées à un usage **éducatif et
personnel** ; toute redistribution est interdite sans autorisation. En
conséquence :

- `scrape_images.py` limite volontairement le débit et la concurrence des
  requêtes (site non commercial, maintenu par deux personnes) ;
- les images téléchargées (`images/`) et les poids entraînés
  (`checkpoints/`) sont destinés à un usage local d'expérimentation — à ne
  **pas** republier ou redistribuer en l'état ;
- la plupart des œuvres reproduites sont dans le domaine public (œuvres
  antérieures à 1900), mais les *reproductions photographiques* elles-mêmes
  restent soumises aux conditions d'usage du site qui les héberge.
