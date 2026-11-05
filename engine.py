"""Moteur local d'Idol : chargement, quantification (INT8 / NF4) et déchargement CPU/GPU pour cartes 6-16 Go.

Le code de génération reprend celui de l'SDK officiel (speridlabs/idol-sdk, Apache-2.0, copié dans ./idol) ;
seule la gestion de la mémoire GPU change.

Stratégie mémoire :
- FP16 : tous les modèles restent en RAM système ; chaque étape (fond, encodeur, denoiser, décodeurs) ne
  met sur le GPU que les modules dont elle a besoin, puis les renvoie en RAM (mode "économique").
- INT8 / NF4 (bitsandbytes) : les blocs transformer du denoiser et de DINOv3 sont quantifiés. Ces modules
  sont petits, donc ils restent en permanence sur le GPU (bitsandbytes ne supporte pas le déplacement
  CPU<->GPU des poids quantifiés). Les décodeurs et BiRefNet restent en FP16 et sont déchargés.
"""
import gc
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from tensordict import TensorDict

from idol.conditioning import DTYPES, EncoderAccessError, ImageEncoder, require_encoder_access
from idol.models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
from idol.pipeline import IdolOutput, IdolPipeline
from idol.sampler import sample_flow

REPO = 'speridlabs/idol'
WEIGHT_FILES = ['config.json', 'denoiser.safetensors', 'decoder.safetensors', 'decoder_gs.safetensors',
                'birefnet.safetensors']
# Blocs transformer du denoiser à quantifier (les embeddings, adaLN et couches de sortie restent en FP16).
DENOISER_QUANT_PREFIXES = ('blocks.', 'noise_refiner.', 'context_refiner.')
# Miroirs non gated, sans compte ni acceptation de conditions. Mêmes poids que les dépôts officiels :
# - LibreYOLO/LibreDINOv3vith16plus : DINOv3 ViT-H+/16 de Meta, converti en bf16 (tenseurs identiques)
# - unsloth/FLUX.2-dev (sous-dossier vae/) : VAE FLUX.2 de Black Forest Labs, non modifié
# Les licences d'origine (DINOv3 License, FLUX.2-dev Non-Commercial License) restent applicables.
MIRRORS = ('LibreYOLO/LibreDINOv3vith16plus', 'unsloth/FLUX.2-dev')

PRECISIONS = {
    'auto': 'Auto (recommandé)',
    'fp16': 'FP16 : qualité maximale',
    'int8': 'INT8 : ~moitié de VRAM, légère perte',
    'nf4': 'NF4 4 bits : le plus léger, perte visible possible',
}


class EngineError(RuntimeError):
    pass


def gpu_info():
    """Infos GPU NVIDIA, sans importer CUDA si la machine n'en a pas."""
    if not torch.cuda.is_available():
        return {'available': False, 'name': None, 'vram_gb': 0.0, 'bf16': False}
    props = torch.cuda.get_device_properties(0)
    return {'available': True, 'name': props.name, 'vram_gb': props.total_memory / 1024 ** 3,
            'bf16': torch.cuda.is_bf16_supported()}


def resolve_auto(info):
    """Auto : FP16 partout. Le déchargement économique s'active sous 12 Go de VRAM."""
    return 'fp16', info['vram_gb'] < 12


def bitsandbytes_available():
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _quantize_denoiser_linears(model: nn.Module, kind: str) -> int:
    """Remplace les nn.Linear des blocs transformer par des couches bitsandbytes (INT8 ou NF4).
    Retourne le nombre de couches quantifiées."""
    import bitsandbytes as bnb

    count = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f'{parent_name}.{child_name}' if parent_name else child_name
            if not isinstance(child, nn.Linear) or not full.startswith(DENOISER_QUANT_PREFIXES):
                continue
            weight = child.weight.data.to(torch.float16).cpu().contiguous()
            has_bias = child.bias is not None
            if kind == 'int8':
                new = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=has_bias,
                                          has_fp16_weights=False, threshold=6.0)
                new.weight = bnb.nn.Int8Params(weight, requires_grad=False, has_fp16_weights=False).to('cuda')
            else:
                new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=has_bias,
                                        compute_dtype=torch.float16, quant_type='nf4', compress_statistics=True)
                new.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type='nf4',
                                               compress_statistics=True).to('cuda')
            if has_bias:
                new.bias = nn.Parameter(child.bias.data.to(torch.float16).to('cuda'), requires_grad=False)
            setattr(parent, child_name, new)
            count += 1
    return count


class _Encoder(ImageEncoder):
    """Encodeur d'image de l'SDK, construit à la main pour accepter DINOv3 déjà quantifié."""

    def _dinov3_tokens(self, images):
        mean = torch.tensor((0.485, 0.456, 0.406), device=images.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=images.device).view(1, 3, 1, 1)
        images = (images.float() - mean) / std
        tokens = self.dinov3(pixel_values=images.to(self.dtype)).last_hidden_state.float()
        return torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])


class LocalIdol(IdolPipeline):
    """Pipeline Idol avec gestion de la mémoire GPU configurable. Ne charge rien tant qu'on ne l'instancie pas."""

    def __init__(self, precision: str = 'fp16', low_vram: bool = False, log=print):
        if not torch.cuda.is_available():
            raise EngineError('Aucun GPU NVIDIA CUDA détecté. Idol nécessite une carte NVIDIA.')
        if precision in ('int8', 'nf4') and not bitsandbytes_available():
            raise EngineError('bitsandbytes n\'est pas installé : INT8/NF4 indisponibles. Utilisez FP16.')
        self.device = torch.device('cuda')
        self.precision = precision
        self.low_vram = low_vram
        self.optimize = False  # torch.compile désactivé : pas de gain garanti sous Windows et au déchargement
        self._pinned = set()   # modules quantifiés : toujours sur le GPU
        self._offload_ok = True
        log(f'Chargement du modèle ({PRECISIONS.get(precision, precision)}, '
            f'{"mémoire économique" if low_vram else "mémoire standard"})...')
        t0 = time.time()

        weights_dir = snapshot_download(REPO, allow_patterns=WEIGHT_FILES)
        with open(os.path.join(weights_dir, 'config.json')) as f:
            self.config = json.load(f)
        dinov3_repo, vae_repo = self._resolve_encoder_repos(self.config['image_encoders'], log)

        def build(model, name):
            model.load_state_dict(load_file(os.path.join(weights_dir, f'{name}.safetensors')), strict=True)
            return model.eval().requires_grad_(False)

        self.denoiser = build(LatentSeqMMFlowModel(**self.config['denoiser']['args']), 'denoiser')
        self.decoder = build(OctreeProbabilityFixedlenDecoder(**self.config['decoder']['args']), 'decoder')
        self.decoder_gs = build(GaussianFixedlenDecoder(**self.config['decoder_gs']['args']), 'decoder_gs')
        self.background_remover = build(BiRefNet(), 'birefnet').half()

        dtype_name = 'bf16' if gpu_info()['bf16'] else 'fp16'  # les GPU sans bf16 (RTX 20xx, GTX 16xx) : fp16
        dtype = DTYPES[dtype_name]

        if precision in ('int8', 'nf4'):
            n = _quantize_denoiser_linears(self.denoiser, precision)
            log(f'Denoiser : {n} couches quantifiées en {precision.upper()}.')
            dinov3 = self._load_dinov3_quantized(dinov3_repo, precision, dtype)
            self._pinned = {self.denoiser, dinov3}
        else:
            dinov3 = self._load_dinov3(dinov3_repo, dtype)
        from diffusers.models import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(vae_repo, subfolder='vae',
                                                 torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        encoder = _Encoder.__new__(_Encoder)  # on évite ImageEncoder.__init__, qui place tout sur le GPU
        encoder.device = self.device
        encoder.dtype = dtype
        encoder.dinov3 = dinov3
        encoder.flux2_vae = vae
        self.image_encoder = encoder

        # placement initial : tout sur le GPU en mémoire standard ; en économique, seuls les modules quantifiés
        # (épinglés) y restent, les autres sont déplacés à la demande par _on_gpu
        everything = [self.background_remover, self.denoiser, self.decoder, self.decoder_gs, dinov3, vae]
        for m in (everything if not low_vram else sorted(self._pinned, key=id)):
            m.to(self.device)
        torch.cuda.empty_cache()
        log(f'Modèle prêt en {time.time() - t0:.0f} s.')

    @staticmethod
    def _resolve_encoder_repos(encoders, log):
        """Dépôts officiels (gated) si le jeton y a accès, sinon miroirs non gated aux mêmes poids.
        IDOL_ENCODER_SOURCE=official|mirror|auto (défaut auto) force le choix."""
        source = os.environ.get('IDOL_ENCODER_SOURCE', 'auto').lower()
        official = (encoders['dinov3'], encoders['flux2_vae'])
        if source == 'mirror':
            log('Encodeurs : miroirs non gated (IDOL_ENCODER_SOURCE=mirror).')
            return MIRRORS
        try:
            require_encoder_access(list(official))
            return official
        except EncoderAccessError:
            if source == 'official':
                raise
        log('Dépôts officiels inaccessibles avec ce jeton : utilisation des miroirs non gated (mêmes poids).')
        return MIRRORS

    def _load_dinov3(self, repo, dtype):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(repo, torch_dtype=dtype)
        return model.eval().requires_grad_(False)

    def _load_dinov3_quantized(self, repo, kind, dtype):
        from transformers import AutoModel, BitsAndBytesConfig
        if kind == 'int8':
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype,
                                       bnb_4bit_use_double_quant=True)
        model = AutoModel.from_pretrained(repo, quantization_config=quant, torch_dtype=dtype,
                                          device_map={'': 0})
        return model.eval().requires_grad_(False)

    @contextmanager
    def _on_gpu(self, *modules):
        """Mode économique : met les modules sur le GPU pour une étape, puis les renvoie en RAM."""
        if not self.low_vram:
            yield
            return
        moved = [m for m in modules if m not in self._pinned]
        for m in moved:
            m.to(self.device)
        try:
            yield
        finally:
            for m in moved:
                m.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, image, mode='synthetic', seed=42, steps=50, cfg_strength=3.0, rescale_t=3.0,
                 num_gaussians=262144, crop_margin=None, verbose=False) -> IdolOutput:
        if isinstance(image, str):
            image = Image.open(image)
        with self._on_gpu(self.background_remover):
            prepared = self.preprocess(image, mode=mode, crop_margin=crop_margin)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(self.device)

        torch.manual_seed(seed)
        with self._on_gpu(self.image_encoder.dinov3, self.image_encoder.flux2_vae):
            cond = self.image_encoder(pixels)
        noise = TensorDict({
            'latent': torch.randn(1, self.denoiser.q_token_length, self.denoiser.in_channels, device=self.device),
            'camera': torch.randn(1, 1, self.denoiser.cam_channels, device=self.device),
        }, batch_size=1, device=self.device)
        with self._on_gpu(self.denoiser):
            sample = sample_flow(self.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                                 steps=steps, rescale_t=rescale_t, cfg_strength=cfg_strength, verbose=verbose,
                                 batch_cfg=False)
        with self._on_gpu(self.decoder, self.decoder_gs):
            gaussian = self.decode(sample['latent'], num_gaussians)
        from idol.camera import decode_camera
        return IdolOutput(gaussian=gaussian, preprocessed_image=prepared,
                          camera=decode_camera(sample['camera'][0, 0]))

    def unload(self):
        for name in ('denoiser', 'decoder', 'decoder_gs', 'background_remover', 'image_encoder'):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()
