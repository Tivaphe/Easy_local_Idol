# Idol en local (Windows & Linux)

Interface web locale pour **Idol** (speridlabs) : une image d'objet en entrée, un objet 3D Gaussian Splatting (`.ply`) en sortie.
L'interface reprend le principe de la [démo officielle](https://huggingface.co/spaces/speridlabs/idol), mais tourne sur votre machine.

## Ce qu'il faut

- Une carte **NVIDIA** avec **6 Go de VRAM minimum** (6 à 16 Go supportés, réglage automatique).
  - Pilote NVIDIA récent. Les cartes RTX 20xx à 50xx sont supportées ; les GTX 10xx (Pascal) ne le sont pas.
  - AMD, Intel et Mac ne sont pas supportés (Idol nécessite CUDA).
- **16 Go de RAM** recommandés, environ **15 Go d'espace disque** (modèles + PyTorch).
- **Python 3.12** (3.10 à 3.12). Windows : [python.org](https://www.python.org/downloads/), cochez *Add python.exe to PATH*.
  Linux : `sudo apt install python3 python3-venv` (Ubuntu / Debian).
- Une connexion Internet pour la première installation (environ 3 Go de modèles, plus PyTorch).

## Installation et lancement

**Windows** : double-cliquez sur `Lancer_Idol.bat`.

**Linux** :
```bash
chmod +x lancer_idol.sh
./lancer_idol.sh
```

Le script crée un environnement `.venv`, installe PyTorch avec CUDA et les dépendances, puis ouvre
**http://127.0.0.1:7860** dans votre navigateur. Les lancements suivants sont immédiats.

### Encodeurs d'image (rien à faire dans la plupart des cas)

Idol lit l'image avec deux encodeurs (DINOv3 de Meta et VAE FLUX.2 de Black Forest Labs). Leurs dépôts officiels
sont « gated » (acceptation de conditions requise). Deux options :

- **Sans rien faire** : si le jeton n'a pas accès aux dépôts officiels, Idol utilise automatiquement des
  **miroirs non gated** aux mêmes poids : [LibreYOLO/LibreDINOv3vith16plus](https://huggingface.co/LibreYOLO/LibreDINOv3vith16plus)
  (DINOv3 ViT-H+/16, converti en bf16) et le sous-dossier `vae/` de [unsloth/FLUX.2-dev](https://huggingface.co/unsloth/FLUX.2-dev).
  Les licences d'origine restent applicables (DINOv3 License ; FLUX.2-dev Non-Commercial License).
- **Dépôts officiels** : acceptez les conditions sur
  https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m et
  https://huggingface.co/black-forest-labs/FLUX.2-dev, créez un jeton (*Settings → Access Tokens*, « Read »),
  collez-le dans **Compte Hugging Face** de l'interface, puis cliquez sur **Vérifier l'accès** (deux ✅).

Pour forcer une source, définissez `IDOL_ENCODER_SOURCE=official` ou `IDOL_ENCODER_SOURCE=mirror` avant le lancement.

## Utilisation

1. Glissez une image (ou choisissez un exemple en bas de page).
2. Choisissez le **type d'entrée** : *Objet synthétique* pour un rendu ou une image générée, *Photo réelle* pour une photo.
3. Cliquez sur **Générer**. La première génération prend plus de temps (chargement du modèle).
4. Faites tourner l'objet dans le visualiseur, téléchargez le `.ply` ou ouvrez-le dans un visualiseur de splats
   (ex. [SuperSplat](https://superspl.at/editor), Blender avec un add-on 3DGS). Les fichiers sont aussi dans `outputs/`.

### Réglages

| Réglage | Rôle |
|---|---|
| **Précision** | Format des poids. *Auto* choisit FP16 (voir ci-dessous). |
| **Mémoire GPU** | *Auto* : mode économique sous 12 Go de VRAM. *Économique* décharge les modules entre les étapes. |
| **Gaussiennes** | Nombre de points 3D formant l'objet. Plus = plus de détails, mais plus de VRAM, de temps et de poids de fichier. 262 144 par défaut. |
| **Étapes** | Étapes de l'algorithme de génération. Plus = qualité légèrement meilleure (marges décroissantes) et temps proportionnellement plus long. 50 par défaut. |
| **Guidage** | Force du guidage par rapport à l'image d'entrée (CFG). 0 = l'image guide à peine · 3 = recommandé · très élevé = sur-respect pouvant créer des artefacts. |
| **Décalage temporel** | Répartit les étapes vers le début de la génération (bruit fort = structure générale de l'objet). 1 = réparti uniformément · 3 = recommandé · 6 = très concentré sur la structure. |
| **Graine** | Aléatoire de départ : mêmes réglages + même graine = exactement le même objet. Changez-la pour obtenir une autre variante. |
| **Graine aléatoire** | Ignore le champ *Graine* et en tire une nouvelle à chaque génération. |

Tous ces réglages (hors *Précision* / *Mémoire GPU*) sont ceux de la démo officielle ; une explication
s'affiche aussi directement sous chaque libellé dans l'interface.

Changer la précision ou le mode mémoire recharge le modèle (environ 30 s).

### Précisions disponibles

| Précision | Poids principaux (estimé) | Qualité | Usage conseillé |
|---|---|---|---|
| **FP16** | ~4 Go | Référence | Carte 8 Go et plus, ou 6 Go en mode économique |
| **INT8** (bitsandbytes) | ~2 Go | Très proche de FP16 | Si FP16 manque de VRAM |
| **NF4 4 bits** (bitsandbytes) | ~1,5 Go | Perte visible possible | Cartes 6 Go ou tâches rapides |

INT8 et NF4 quantifient le denoiser et l'encodeur DINOv3 (les décodeurs restent en FP16, pour la géométrie).
La quantification se fait au chargement : aucun fichier supplémentaire à télécharger.

Si une génération échoue avec « mémoire GPU insuffisante » : passez en **INT8**, puis en **NF4**, ou
réduisez les Gaussiennes à 131 072.

## Test rapide (smoke test)

Pour valider votre configuration — notamment les modes rarement testés (mémoire *économique*, INT8, NF4) :

```bash
# Linux (dossier du dépôt, après un premier lancement)
.venv/bin/python smoke_test.py --all

# Windows (invite de commandes dans le dossier du dépôt)
.venv\Scripts\python smoke_test.py --all
```

Chaque combinaison charge le modèle, génère un vrai objet (65 536 gaussiennes, 10 étapes), écrit le `.ply`
dans `outputs/smoke/` puis décharge le modèle. Le résultat s'affiche en **PASS / FAIL** avec la durée et la
pointe de VRAM ; le code retour est non nul en cas d'échec. Sans `--all`, seule la combinaison demandée est
testée (`--precision fp16|int8|nf4`, `--memory standard|economy`).

## Structure du dossier

```
Lancer_Idol.bat / lancer_idol.sh   point d'entrée (Windows / Linux)
launcher.py                        installation (PyTorch CUDA, dépendances) et lancement
app.py                             interface web (Gradio)
presets.py                         liste des exemples (presets.json, ou génération auto s'il manque)
engine.py                          chargement, quantification, gestion de la mémoire GPU
smoke_test.py                      test rapide des combinaisons précision × mémoire
idol/                              code de l'SDK officiel (Apache-2.0, voir idol/LICENSE-IDOL-SDK)
                                   (seule modification locale : callback de progression optionnel dans sampler.py)
examples/                          images d'exemple de la démo officielle (+ presets.json)
outputs/                           fichiers .ply générés
LICENSE                            licence Apache-2.0 du dépôt
.github/workflows/ci.yml           intégration continue (syntaxe, presets, shellcheck)
```

## Dépannage

- **« Aucun GPU NVIDIA détecté »** : installez le pilote NVIDIA, puis relancez. Vérifiez avec `nvidia-smi`.
- **« PyTorch ne voit pas le GPU »** : mettez à jour le pilote NVIDIA, supprimez le dossier `.venv` et relancez.
- **« Accès refusé » à un encodeur** : Idol bascule normalement sur les miroirs. Pour les dépôts officiels, acceptez leurs conditions et vérifiez le jeton.
- **INT8 / NF4 indisponibles** : bitsandbytes n'a pas pu être installé ; FP16 reste disponible. Relancez le script.
- **Erreur « requires accelerate »** : dépendance manquante dans un `.venv` créé avant sa mise à jour —
  relancez `Lancer_Idol.bat` / `lancer_idol.sh` (l'installation se met à jour automatiquement), ou
  `.venv\Scripts\pip install accelerate` (Windows) / `.venv/bin/pip install accelerate` (Linux).
- **Port 7860 occupé** : `IDOL_PORT=7870` (Linux) ou définissez la variable d'environnement avant de lancer `app.py`.

## Limites honnêtes

- Le code de gestion mémoire (déchargement, INT8/NF4) n'a pas encore été validé sur toutes les cartes NVIDIA.
  Lancez `smoke_test.py --all` (voir *Test rapide*) pour vérifier le vôtre ; si une étape échoue, copiez le
  message d'erreur complet.
- Les chiffres de VRAM sont des estimations (le SDK annonce 7,2 Go de pic sans déchargement).
- La quantification INT8 / NF4 est faite par l'application. Il n'existe pas aujourd'hui de version quantifiée officielle
  pour CUDA : les variantes publiées sur Hugging Face sont pour Apple (MLX) ou le navigateur (WebGPU).
- `torch.compile` (accélération d'environ 1,7x du SDK) est désactivé pour rester stable sous Windows.
- Idol est un modèle de recherche : la qualité varie selon les images (objets isolés sur fond simple = meilleurs résultats).

## Crédits

- [speridlabs/idol](https://huggingface.co/spaces/speridlabs/idol) — modèle Idol et démo officielle.
  Le code du dossier `idol/` est le SDK officiel (Apache-2.0, © speridlabs), voir `idol/LICENSE-IDOL-SDK`.
- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) — encodeur DINOv3 (© Meta, licence DINOv3).
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) — encodeur VAE FLUX.2 (© Black Forest Labs, licence non commerciale).
- Miroirs non gated utilisés par défaut : [LibreYOLO/LibreDINOv3vith16plus](https://huggingface.co/LibreYOLO/LibreDINOv3vith16plus),
  [unsloth/FLUX.2-dev](https://huggingface.co/unsloth/FLUX.2-dev) (sous-dossier `vae/`).
- Images d'`examples/` issues de la démo officielle.
- [Gradio](https://www.gradio.app), [PyTorch](https://pytorch.org), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — dépendances principales.

Les poids de modèle ont leurs propres licences (dont une licence non commerciale pour FLUX.2-dev) : elles
s'appliquent indépendamment de la licence de ce code.

## Licence

Le code de ce dépôt (interface, moteur, lanceurs, scripts) est sous **Apache-2.0**, voir le fichier `LICENSE`.
Le code d'Idol (`idol/`) est le SDK officiel, sous Apache-2.0, copyright speridlabs, voir `idol/LICENSE-IDOL-SDK`.
Les poids et les encodeurs ont leurs propres licences, à consulter sur leurs pages Hugging Face (voir *Crédits*).
