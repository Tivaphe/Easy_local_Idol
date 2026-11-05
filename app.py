"""Idol en local : interface web (même esprit que l'espace Hugging Face speridlabs/idol).

Lancement : via launcher.py (Windows : Lancer_Idol.bat, Linux : lancer_idol.sh), ou directement :
    python app.py
"""
import gc
import os
import random
import threading
import time
from pathlib import Path

import gradio as gr
import torch

from engine import PRECISIONS, EngineError, LocalIdol, bitsandbytes_available, gpu_info, resolve_auto
from presets import load_presets

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
PRESETS = load_presets(APP_DIR)

MODES = {'Objet synthétique (rendu, image générée)': 'synthetic', 'Photo réelle': 'real'}
MODE_LABEL = {v: k for k, v in MODES.items()}
GAUSSIANS = {'65 536': 65536, '131 072': 131072, '262 144': 262144, '524 288': 524288, '1 048 576': 1048576}
GAUSSIAN_LABEL = {v: k for k, v in GAUSSIANS.items()}
PRECISION_CHOICES = [(label, key) for key, label in PRECISIONS.items()]
MEMORY_CHOICES = [('Auto', 'auto'), ('Standard (tout sur le GPU)', 'standard'),
                  ('Économique (6-12 Go)', 'economy')]
HF_GATED = ['https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m',
            'https://huggingface.co/black-forest-labs/FLUX.2-dev']


class Manager:
    """Garde un seul modèle chargé à la fois ; le recharge seulement si précision ou mémoire changent."""

    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.key = None

    def describe(self):
        if self.engine is None:
            return 'Aucun modèle chargé (il se chargera au premier clic sur Générer).'
        prec = PRECISIONS.get(self.key[0], self.key[0]).split(' :')[0]
        mem = 'économique' if self.key[1] else 'standard'
        return f'Modèle chargé : **{prec}**, mémoire **{mem}**.'

    def get(self, precision, low_vram, log):
        key = (precision, low_vram)
        if self.engine is not None and self.key == key:
            return self.engine
        if self.engine is not None:
            log('Déchargement du modèle précédent...')
            self.engine.unload()
            self.engine = None
            self.key = None
            gc.collect()
            torch.cuda.empty_cache()
        try:
            self.engine = LocalIdol(precision=precision, low_vram=low_vram, log=log)
        except BaseException:
            # Chargement échoué (réseau, OOM...) : rien ne doit rester en VRAM.
            gc.collect()
            torch.cuda.empty_cache()
            raise
        self.key = key
        return self.engine


MANAGER = Manager()


def hardware_text():
    info = gpu_info()
    if not info['available']:
        return ('⚠️ **Aucun GPU NVIDIA détecté.** Idol a besoin d\'une carte NVIDIA (CUDA) avec 6 Go de VRAM ou plus.')
    prec, low = resolve_auto(info)
    lines = [f'**GPU :** {info["name"]} · {info["vram_gb"]:.1f} Go de VRAM',
             f'**Réglage Auto :** {PRECISIONS[prec].split(" :")[0]}, '
             f'mémoire {"économique" if low else "standard"}']
    if not bitsandbytes_available():
        lines.append('ℹ️ INT8 / NF4 indisponibles (bitsandbytes non installé).')
    return '  \n'.join(lines)


def resolve_settings(precision, memory):
    info = gpu_info()
    auto_prec, auto_low = resolve_auto(info) if info['available'] else ('fp16', True)
    prec = auto_prec if precision == 'auto' else precision
    if memory == 'auto':
        low = auto_low
    else:
        low = memory == 'economy'
    return prec, low


def check_access():
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    lines = []
    for url in HF_GATED:
        repo = url.replace('https://huggingface.co/', '')
        try:
            auth_check(repo)
            lines.append(f'✅ Accès OK : `{repo}`')
        except (GatedRepoError, RepositoryNotFoundError):
            lines.append(f'❌ Accès refusé : `{repo}`. Acceptez les conditions sur {url}, ou laissez Idol utiliser le miroir non gated.')
        except Exception as e:  # réseau, token invalide, etc.
            lines.append(f'⚠️ Vérification impossible pour `{repo}` ({type(e).__name__}).')
    return '  \n'.join(lines)


def save_token(token):
    from huggingface_hub import login
    token = (token or '').strip()
    if not token:
        return 'Collez un jeton Hugging Face (commence par `hf_`).'
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        return f'Jeton refusé : {e}'
    return 'Jeton enregistré. Cliquez sur « Vérifier l\'accès ».'


def generate(image, input_type, steps, cfg_strength, rescale_t, seed, randomize, gaussians, precision, memory,
             progress=gr.Progress()):
    if image is None:
        raise gr.Error('Ajoutez une image ou choisissez un exemple.')
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    steps = int(steps)
    prec, low = resolve_settings(precision, memory)
    logs = []

    def log(msg):
        logs.append(msg)
        progress(state['frac'], desc=msg)

    state = {'frac': 0.0}
    t0 = time.time()
    with MANAGER.lock:
        try:
            engine = MANAGER.get(prec, low, log)
            state['frac'] = 0.4
            progress(0.4, desc='Modèle prêt, génération...')

            def on_step(i, n):
                state['frac'] = 0.45 + 0.45 * (i + 1) / n
                progress(state['frac'], desc=f'Étape {i + 1} / {n}')

            output = engine(image, mode=MODES.get(input_type, input_type), seed=seed, steps=steps,
                            cfg_strength=cfg_strength, rescale_t=rescale_t,
                            num_gaussians=GAUSSIANS.get(gaussians, gaussians), on_step=on_step)
        except EngineError as e:
            raise gr.Error(str(e))
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            raise gr.Error('Mémoire GPU insuffisante (chargement ou génération). Essayez la précision INT8 ou '
                           'NF4, passez la mémoire en « Économique », ou réduisez le nombre de Gaussiennes.')
        ply = OUTPUT_DIR / f'idol-{time.strftime("%Y%m%d-%H%M%S")}-seed{seed}.ply'
        output.gaussian.save_ply(str(ply))
    elapsed = time.time() - t0
    info = (f'<div class="sl-meta"><span>{GAUSSIAN_LABEL.get(GAUSSIANS.get(gaussians, 0), gaussians)} Gaussiennes · '
            f'{steps} étapes · graine {seed} · {elapsed:.0f} s · {PRECISIONS[prec].split(" :")[0]}'
            f'{" · mémoire économique" if low else ""}</span></div>')
    return str(ply), str(ply), output.preprocessed_image, seed, info, MANAGER.describe()


def pick_example(evt: gr.SelectData):
    p = PRESETS[evt.index]
    return (str(APP_DIR / p['image']), MODE_LABEL[p['mode']], p['steps'], p['cfg'], p['rescale_t'], p['seed'], False,
            GAUSSIAN_LABEL[p['num_gaussians']])


THEME = gr.themes.Base(font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
                       font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'monospace'],
                       radius_size=gr.themes.sizes.radius_none).set(
    button_primary_background_fill='#000000', button_primary_background_fill_hover='#d94a1f',
    button_primary_text_color='#ffffff', color_accent='#d94a1f', slider_color='#d94a1f', block_shadow='none')
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }
.sl-head { padding: 28px 0 20px; border-bottom: 1px solid #e3e3e3; margin-bottom: 8px; }
.sl-head .sl-label { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: #6b6b6b; }
.sl-head h1 { margin: 10px 0 0; font-size: clamp(48px, 6vw, 88px); font-weight: 700; line-height: .9;
  letter-spacing: -0.05em; }
.sl-head p { margin: 14px 0 0; max-width: 62ch; font-size: 17px; line-height: 1.5; color: #444; }
.sl-head p em { color: #d94a1f; font-style: normal; font-weight: 600; }
.sl-links { display: flex; gap: 10px; margin-top: 16px; }
.sl-links a { border: 1px solid #000; padding: 6px 14px; color: #000 !important; text-decoration: none; font-size: 14px; }
.sl-links a:hover { background: #000; color: #fff !important; }
#generate { height: 52px; font-size: 17px; font-weight: 600; }
.sl-meta { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #444; padding-top: 6px; }
.sl-section { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
  color: #6b6b6b; border-top: 1px solid #e3e3e3; padding-top: 24px; margin-top: 24px; }
#presets .thumbnail-item { border: 0 !important; box-shadow: none !important; }
#presets .caption-label { display: none; }
"""
HEAD = """
<div class="sl-head">
  <span class="sl-label">Speridlabs · Idol · version locale</span>
  <h1>Idol</h1>
  <p><strong>Une image en entrée, un objet 3D en sortie.</strong> Idol transforme une photo ou un rendu d'objet
  en 3D Gaussian Splatting, y compris les faces que l'image ne montre pas. <em>Tout tourne sur votre GPU.</em></p>
  <div class="sl-links">
    <a href="https://github.com/speridlabs/idol-sdk" target="_blank">Code</a>
    <a href="https://huggingface.co/speridlabs/idol" target="_blank">Modèle</a>
    <a href="https://huggingface.co/spaces/speridlabs/idol" target="_blank">Démo officielle</a>
  </div>
</div>
"""

with gr.Blocks(title='Idol · local', theme=THEME, css=CSS) as demo:
    gr.HTML(HEAD)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(value=str(APP_DIR / PRESETS[0]['image']), type='pil', image_mode='RGBA',
                             label='Image d\'entrée', sources=['upload', 'clipboard'], height=340)
            input_type = gr.Radio(list(MODES), value=MODE_LABEL[PRESETS[0]['mode']], label='Type d\'entrée')
            run_btn = gr.Button('Générer', variant='primary', elem_id='generate')
            with gr.Accordion('Paramètres', open=False):
                gaussians = gr.Dropdown(list(GAUSSIANS), value='262 144', label='Gaussiennes')
                steps = gr.Slider(10, 100, value=50, step=1, label='Étapes')
                cfg_strength = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label='Guidage')
                rescale_t = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label='Décalage temporel')
                seed = gr.Number(value=42, precision=0, label='Graine')
                randomize = gr.Checkbox(value=False, label='Graine aléatoire')
            with gr.Accordion('Modèle et mémoire GPU', open=False):
                precision = gr.Dropdown(PRECISION_CHOICES, value='auto', label='Précision (quantification)')
                memory = gr.Radio(MEMORY_CHOICES, value='auto', label='Mémoire GPU')
                gr.Markdown(hardware_text())
                status = gr.Markdown(MANAGER.describe())
                gr.Markdown('Changer la précision ou la mémoire recharge le modèle (environ 30 s). '
                            'INT8 et NF4 réduisent la VRAM, mais la qualité peut légèrement baisser.')
            with gr.Accordion('Compte Hugging Face', open=False):
                gr.Markdown('Idol utilise deux encodeurs. Sans accès officiel, Idol charge automatiquement des miroirs non gated '
                            '(mêmes poids, licences d\'origine). Pour les dépôts officiels : acceptez leurs conditions, '
                            'puis collez votre jeton (Settings → Access Tokens, lecture seule suffit).  \n'
                            + '  \n'.join(f'- {u}' for u in HF_GATED))
                token = gr.Textbox(label='Jeton Hugging Face', type='password', placeholder='hf_...')
                with gr.Row():
                    save_btn = gr.Button('Enregistrer le jeton')
                    check_btn = gr.Button('Vérifier l\'accès')
                access_md = gr.Markdown()
        with gr.Column(scale=7):
            result = gr.Model3D(label='Objet 3D', height=560, clear_color=(0.05, 0.05, 0.05, 1.0))
            download = gr.File(label='Fichier .ply (à ouvrir dans un visualiseur de splats)')
            prepared = gr.Image(label='Image vue par le modèle', height=180, type='pil')
            info = gr.HTML('')
    gr.HTML('<div class="sl-section">Exemples : cliquez pour charger l\'image et ses paramètres</div>')
    gallery = gr.Gallery(value=[str(APP_DIR / p['image']) for p in PRESETS], columns=8, height='auto',
                         allow_preview=False, show_label=False, elem_id='presets', show_download_button=False)

    run_btn.click(generate, [image, input_type, steps, cfg_strength, rescale_t, seed, randomize, gaussians,
                             precision, memory],
                  [result, download, prepared, seed, info, status], show_progress_on=result)
    gallery.select(pick_example, None, [image, input_type, steps, cfg_strength, rescale_t, seed, randomize,
                                        gaussians])
    save_btn.click(save_token, [token], [access_md])
    check_btn.click(check_access, None, [access_md])

if __name__ == '__main__':
    port = int(os.environ.get('IDOL_PORT', '7860'))
    demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=port,
                                  allowed_paths=[str(APP_DIR)],
                                  inbrowser=os.environ.get('IDOL_NO_BROWSER') != '1')
