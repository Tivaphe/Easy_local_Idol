"""Installation et lancement automatiques d'Idol (Windows et Linux).

Appelé par Lancer_Idol.bat (Windows) ou lancer_idol.sh (Linux). Il :
1. crée un environnement virtuel (.venv) dans ce dossier ;
2. installe PyTorch avec CUDA (le paquet PyPI par défaut est CPU seulement) ;
3. installe les dépendances et bitsandbytes (INT8/NF4, facultatif) ;
4. lance l'interface web sur http://127.0.0.1:7860.
Relancer le script ne réinstalle que ce qui a changé.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
VENV = APP_DIR / '.venv'
REQUIREMENTS = APP_DIR / 'requirements.txt'
CONSTRAINTS = APP_DIR / 'constraints.txt'
STAMP = VENV / '.idol-requirements-v2.sha256'  # v2 : installation avec contraintes torch
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'  # CUDA 12.8 : RTX 20xx à RTX 50xx
TORCH_PINS = ['torch==2.8.0', 'torchvision==0.23.0']
BNB_PIN = 'bitsandbytes>=0.46.0'


def say(msg):
    print(f'[Idol] {msg}', flush=True)


def fail(msg):
    say(f'ERREUR : {msg}')
    sys.exit(1)


def venv_python():
    return VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], **kw)


def check_python():
    v = sys.version_info
    if v < (3, 10):
        fail(f'Python 3.10 à 3.12 est requis (vous avez {v.major}.{v.minor}).')
    if v >= (3, 13):
        say(f'Attention : Python {v.major}.{v.minor} n\'est pas encore testé. Python 3.12 est recommandé.')


def check_gpu():
    if shutil.which('nvidia-smi') is None:
        fail('Aucun GPU NVIDIA détecté (nvidia-smi introuvable). Idol nécessite une carte NVIDIA avec pilote '
             'installé : https://www.nvidia.com/Download/index.aspx')
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        say(f'GPU détecté : {out.stdout.strip().splitlines()[0]}')
    else:
        fail('nvidia-smi ne répond pas. Vérifiez le pilote NVIDIA.')


def ensure_venv():
    if not venv_python().exists():
        say('Création de l\'environnement Python (.venv)...')
        run([sys.executable, '-m', 'venv', VENV], check=True)
    run([venv_python(), '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'])


def torch_ok():
    """PyTorch CUDA présent ET torchvision compatible (ses opérateurs compilés doivent se charger)."""
    code = ('import torch, torchvision\n'
            'from torchvision.ops import deform_conv2d, nms\n'
            'import sys; sys.exit(0 if torch.cuda.is_available() else 1)')
    return run([venv_python(), '-c', code], capture_output=True).returncode == 0


def ensure_torch():
    if torch_ok():
        return
    say('Installation de PyTorch et torchvision avec CUDA (plusieurs centaines de Mo, patience)...')
    # --force-reinstall : corrige un torchvision resté sur une autre version que torch
    r = run([venv_python(), '-m', 'pip', 'install', '--force-reinstall', *TORCH_PINS,
             '--index-url', TORCH_INDEX])
    if r.returncode != 0:
        fail('Installation de PyTorch CUDA échouée. Vérifiez votre connexion Internet.')
    if not torch_ok():
        fail('PyTorch est installé mais ne fonctionne pas avec le GPU. Mettez à jour le pilote NVIDIA, '
             'ou supprimez le dossier .venv et relancez.')


def ensure_requirements():
    digest = hashlib.sha256((REQUIREMENTS.read_bytes() + CONSTRAINTS.read_bytes())).hexdigest()
    if STAMP.exists() and STAMP.read_text().strip() == digest:
        return
    say('Installation des dépendances...')
    # -c : empêche pip de remplacer torch/torchvision par une autre version
    r = run([venv_python(), '-m', 'pip', 'install', '-r', REQUIREMENTS, '-c', CONSTRAINTS])
    if r.returncode != 0:
        fail('Installation des dépendances échouée (voir les messages ci-dessus).')
    STAMP.write_text(digest)


def ensure_bitsandbytes():
    r = run([venv_python(), '-c', 'import bitsandbytes'], capture_output=True)
    if r.returncode == 0:
        return
    say('Installation de bitsandbytes (pour INT8 / NF4)...')
    r = run([venv_python(), '-m', 'pip', 'install', BNB_PIN, '-c', CONSTRAINTS])
    if r.returncode != 0:
        say('bitsandbytes n\'a pas pu être installé : seule la précision FP16 sera disponible.')


def main():
    check_python()
    check_gpu()
    ensure_venv()
    ensure_torch()
    ensure_requirements()
    ensure_bitsandbytes()
    ensure_torch()  # dernière vérification : aucune dépendance ne doit avoir cassé torch
    say('Lancement de l\'interface. Ouverture du navigateur sur http://127.0.0.1:7860 ')
    say('Pour arrêter Idol : fermez cette fenêtre ou appuyez sur Ctrl+C.')
    try:
        sys.exit(run([venv_python(), APP_DIR / 'app.py'], cwd=APP_DIR).returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()
