# RVM — Interface Graphique pour Robust Video Matting

Outil perso pour faire de la rotoscopie avec [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) sans avoir à ouvrir un terminal à chaque fois. Interface Gradio locale, tournant uniquement sur **macOS** avec accélération **Apple Silicon (MPS)** — construite notamment pour survivre sur un MacBook Pro M1 8 Go qui rame un peu avec le code d'origine.

---

## Lancement et arrêt

Deux fichiers `.command` à double-cliquer, rien d'autre.

| Action | Fichier |
|---|---|
| Démarrer | **`Lancer RVM.command`** |
| Arrêter | **`Arrêter RVM.command`** |

**Lancer RVM.command** ouvre un Terminal, démarre le serveur Gradio en arrière-plan, puis ouvre automatiquement `http://localhost:7860` dans le navigateur. Une notification macOS confirme quand l'interface est prête. Si le serveur est déjà en cours d'exécution, le script rouvre simplement le navigateur sans rien relancer.

Une fois le navigateur ouvert, vous pouvez fermer la fenêtre Terminal — le serveur continue de tourner.

**Arrêter RVM.command** envoie un signal d'arrêt au processus (SIGTERM puis SIGKILL si besoin), supprime le PID file et envoie une notification macOS. Si le PID file est absent ou périmé, il cherche le processus par nom comme secours.

---

## Installation (première fois)

```bash
git clone <ce-repo>
cd RobustVideoMatting
pip install -r requirements.txt
```

Ensuite, rendre les `.command` exécutables si nécessaire :
```bash
chmod +x "Lancer RVM.command" "Arrêter RVM.command"
```

---

## Poids du modèle

Deux options :

**Option 1 — Depuis l'interface** : au premier lancement, cliquez sur "Télécharger les poids" dans la Section 1.

**Option 2 — Manuel** :
```bash
mkdir -p models
# MobileNetV3 (~14 Mo, rapide)
curl -L "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth" -o models/rvm_mobilenetv3.pth
# ResNet50 (~50 Mo, qualité max)
curl -L "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth" -o models/rvm_resnet50.pth
```

---

## Fonctionnalités

| Section | Contenu |
|---|---|
| **1 — Environnement** | Détection device (MPS/CPU), statut des poids, téléchargement automatique depuis GitHub |
| **2 — Import** | Drag & drop MP4/MOV/MKV/AVI, miniature première frame + métadonnées (résolution, FPS, durée, codec, taille) |
| **3 — Réglages** | Modèle, résolution de traitement, seq_chunk (parallélisme), format de sortie, sorties à générer, fond de prévisualisation, dossier de sortie (auto-suggéré sur le Bureau) |
| **4 — Lancement** | Progression temps réel, bouton Annuler (frames déjà traitées conservées), notification macOS à la fin, bouton "Ouvrir dans le Finder" |
| **5 — Prévisualisation** | Comparaison source/résultat côte à côte à 10%, 25%, 50%, 75% — la source s'affiche dès l'import, le résultat apparaît après l'inférence |

### Sorties disponibles

| Type | Description |
|---|---|
| **Composition finale** | Sujet détouré sur le fond choisi (noir, blanc, vert chroma, damier) |
| **Alpha mask** | Masque en niveaux de gris — utile pour After Effects / DaVinci |
| **Foreground brut** | RGB du sujet seul, sans fond |

Chaque sortie peut être exportée en **MP4**, **séquence PNG**, ou les deux.

---

## Accélération matérielle

- **Apple Silicon M1/M2/M3/M4** : MPS activé automatiquement + fallback CPU pour les ops non supportées (`PYTORCH_ENABLE_MPS_FALLBACK=1` défini automatiquement)
- **CPU** : fonctionne mais plus lent (~2–5 FPS selon la résolution)

---

## Optimisations pour Mac 8 Go

Sur les Mac avec 8 Go de RAM unifiée, l'inférence RVM telle que publiée par l'auteur original montrait une dégradation progressive : 2.4 FPS à la frame 12, 0.9 FPS à la frame 36. Ce fork corrige les quatre causes.

**Lecture vidéo en O(N²).** Le `Dataset` d'origine ouvrait un `VideoCapture` séparé et sautait à la frame N pour chaque appel — soit ~80 000 seeks cumulés pour 400 frames. Remplacé par un `IterableDataset` qui lit séquentiellement.

**28 opérations sans support MPS natif.** MobileNetV3-Large utilise Hardswish et Hardsigmoid, deux ops absentes du backend MPS de PyTorch 2.0.x. Chacune forçait un aller-retour MPS→CPU→MPS, soit 28 par forward pass. Concrètement : la première batch prenait 0.8s, la dixième prenait 25s, et ça ne remontait plus. Tous ces modules sont patchés à chaud avec des équivalents `relu6` — même calcul, aucun fallback déclenché.

**API de vidage du cache cassée dans PyTorch 2.0.1.** `torch.mps.empty_cache()` n'existe pas dans cette version (l'API a changé de nom dans 2.1). L'ancien code lançait une exception silencieuse à la frame 50. Un shim détecte le bon appel selon la version installée.

**Tenseurs MPS maintenus pendant l'écriture disque.** Garder `fgr`, `pha` et `com` (~95 Mo chacun à 1080p) en mémoire pendant l'écriture sur disque fragmentait l'allocateur MPS et déclenchait du swap sur 8 Go. L'ordre est inversé : `del` + vidage du cache avant tout accès disque.

Résultat mesuré sur M1 8 Go, 399 frames 1920×1080 : 54s total (7.4 FPS stable) contre 425s (~0.9 FPS en chute continue) avec le code d'origine.

---

## Architecture

```
RobustVideoMatting/
├── Lancer RVM.command  # Double-clic pour démarrer
├── Arrêter RVM.command # Double-clic pour stopper
├── app.py              # Interface Gradio (5 sections)
├── rvm_inference.py    # Boucle d'inférence isolée (testable en CLI)
├── utils.py            # Helpers macOS (device, vidéo, notif, téléchargement)
├── rvm_model/          # Fichiers modèle RVM (copiés depuis le repo officiel)
│   ├── model.py
│   ├── mobilenetv3.py
│   ├── resnet.py
│   └── ...
├── models/             # Poids .pth (gitignorés)
└── requirements.txt
```

---

## Test CLI

Sans passer par l'interface, `rvm_inference.py` est utilisable directement :

```bash
python rvm_inference.py \
  --input ma_video.mp4 \
  --output ./sortie \
  --backbone mobilenetv3 \
  --seq-chunk 12 \
  --outputs composite,alpha \
  --format video \
  --bg black
```

---

## Compatibilité

- macOS 12+ (Monterey)
- Python 3.9+ (testé sur 3.9.13 via pyenv)
- PyTorch 2.0+ (MPS inclus dans les builds officiels Apple Silicon)
