"""Smoke test : valide le moteur (précision × mémoire) sur une vraie génération courte.

Usage (dans l'environnement .venv) :
    python smoke_test.py                     # réglage de base : FP16, mémoire standard
    python smoke_test.py --precision int8 --memory economy
    python smoke_test.py --all               # toutes les combinaisons supportées

Chaque combinaison charge le modèle, génère un objet (65 536 gaussiennes, 10 étapes),
écrit le .ply dans outputs/smoke/ puis décharge le modèle. Affiche PASS / FAIL avec
la durée et la pointe de VRAM. Code retour : 0 = tout OK, 1 = échec, 2 = pas de GPU.

Objectif : valider les modes rarement testés (économie, INT8, NF4) en quelques minutes.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# (précision, mémoire) testées par --all ; int8/nf4 sont sautées si bitsandbytes absent.
ALL_COMBOS = [('fp16', 'standard'), ('fp16', 'economy'),
              ('int8', 'standard'), ('int8', 'economy'),
              ('nf4', 'standard'), ('nf4', 'economy')]


def parse_args():
    p = argparse.ArgumentParser(description='Smoke test du moteur Idol local.')
    p.add_argument('--precision', default='fp16', choices=['fp16', 'int8', 'nf4'],
                   help='précision à tester (défaut fp16)')
    p.add_argument('--memory', default='standard', choices=['standard', 'economy'],
                   help='mode mémoire à tester (défaut standard)')
    p.add_argument('--all', action='store_true', help='teste toutes les combinaisons supportées')
    p.add_argument('--steps', type=int, default=10, help="nombre d'étapes (défaut 10)")
    p.add_argument('--gaussians', type=int, default=65536, help='nombre de Gaussiennes (défaut 65536)')
    return p.parse_args()


def run_one(precision, memory, image, steps, gaussians):
    import torch
    from engine import LocalIdol

    low = memory == 'economy'
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    t0 = time.time()
    engine = LocalIdol(precision=precision, low_vram=low, log=print)
    t_load = time.time() - t0
    try:
        t1 = time.time()
        output = engine(image, mode='synthetic', seed=0, steps=steps, num_gaussians=gaussians)
        t_gen = time.time() - t1
        out = APP_DIR / 'outputs' / 'smoke' / f'{precision}-{memory}.ply'
        out.parent.mkdir(parents=True, exist_ok=True)
        output.gaussian.save_ply(str(out))
    finally:
        engine.unload()
        gc.collect()
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return t_load, t_gen, peak


def main():
    args = parse_args()

    import torch
    if not torch.cuda.is_available():
        print('SKIP : aucun GPU NVIDIA CUDA détecté (smoke test nécessite CUDA).')
        return 2

    from engine import bitsandbytes_available
    from presets import load_presets

    image = str(APP_DIR / load_presets(APP_DIR)[0]['image'])
    combos = ALL_COMBOS if args.all else [(args.precision, args.memory)]

    print(f'Image : {image}')
    print(f'Génération : {args.gaussians} gaussiennes, {args.steps} étapes\n')

    results = []
    for precision, memory in combos:
        if precision in ('int8', 'nf4') and not bitsandbytes_available():
            print(f'SKIP  {precision:<5} × {memory:<9} : bitsandbytes non installé')
            results.append((precision, memory, 'SKIP', None))
            continue
        label = f'{precision:<5} × {memory:<9}'
        try:
            t_load, t_gen, peak = run_one(precision, memory, image, args.steps, args.gaussians)
            print(f'PASS  {label} : chargement {t_load:.0f} s · génération {t_gen:.0f} s · '
                  f'pointe {peak:.1f} Go de VRAM\n')
            results.append((precision, memory, 'PASS', peak))
        except Exception as e:
            print(f'FAIL  {label} : {type(e).__name__}: {e}\n')
            results.append((precision, memory, 'FAIL', None))
            gc.collect()
            torch.cuda.empty_cache()

    print('--- Résumé ---')
    for precision, memory, status, peak in results:
        extra = f' · {peak:.1f} Go' if peak else ''
        print(f'  {status:<5} {precision} × {memory}{extra}')
    failed = sum(1 for r in results if r[2] == 'FAIL')
    if failed:
        print(f'{failed} combinaison(s) en échec.')
        return 1
    print('Toutes les combinaisons testées passent.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
