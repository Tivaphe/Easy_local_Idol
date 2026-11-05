"""Liste des exemples affichés dans l'interface (aucune dépendance lourde).

Lit ``examples/presets.json`` s'il existe et est valide ; sinon génère une liste
par défaut à partir des images du dossier ``examples/``. L'application démarre
donc toujours, même si presets.json n'a pas été poussé dans le dépôt.
"""
import json
from pathlib import Path

REQUIRED = ('image', 'mode', 'steps', 'cfg', 'rescale_t', 'seed', 'num_gaussians')
DEFAULTS = {'steps': 50, 'cfg': 3.0, 'rescale_t': 3.0, 'seed': 42, 'num_gaussians': 262144}
IMAGE_SUFFIXES = ('.webp', '.png', '.jpg', '.jpeg')


def _fallback(app_dir: Path):
    """Présets par défaut : toutes les images d'exemple, réglages par défaut."""
    examples = app_dir / 'examples'
    presets = []
    if examples.is_dir():
        for path in sorted(examples.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                presets.append({'image': f'examples/{path.name}', 'mode': 'synthetic', **DEFAULTS})
    return presets


def load_presets(app_dir):
    """Retourne une liste de présets valide ; lève FileNotFoundError si aucune image."""
    app_dir = Path(app_dir)
    path = app_dir / 'examples' / 'presets.json'
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            presets = []
            for item in raw:
                if not isinstance(item, dict) or not all(k in item for k in REQUIRED):
                    continue
                item = dict(item)
                item['num_gaussians'] = int(item['num_gaussians'])
                presets.append(item)
            if presets:
                return presets
            print('[Idol] presets.json invalide (clés manquantes) : liste par défaut utilisée.')
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            print(f'[Idol] presets.json illisible ({e}) : liste par défaut utilisée.')
    else:
        print('[Idol] presets.json absent : liste d\'exemples générée depuis examples/.')
    presets = _fallback(app_dir)
    if not presets:
        raise FileNotFoundError(f'Aucune image d\'exemple dans {app_dir / "examples"} '
                                f'et presets.json absent ou invalide.')
    return presets
