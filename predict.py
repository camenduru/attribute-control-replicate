import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/attribute-control')
os.chdir('/content/attribute-control')

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from attribute_control import EmbeddingDelta
from attribute_control.model import SDXL
from attribute_control.prompt_utils import get_mask, get_mask_regex

torch.set_float32_matmul_precision('high')
DEVICE = 'cuda:0'
DTYPE = torch.float16

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = SDXL(
            pipeline_type='diffusers.StableDiffusionXLPipeline',
            model_name='stabilityai/stable-diffusion-xl-base-1.0',
            pipe_kwargs={ 'torch_dtype': DTYPE, 'variant': 'fp16', 'use_safetensors': True },
            device=DEVICE
        )
    def predict(
        self,
        prompt: str = Input(default="a photo of a beautiful man"),
        pretrained_deltas: str = Input(choices=["person_age",
                                            "person_bald",
                                            "person_colorful_outfit",
                                            "person_curly_hair",
                                            "person_elegant",
                                            "person_fitness",
                                            "person_freckled",
                                            "person_groomed",
                                            "person_height",
                                            "person_long_hair",
                                            "person_makeup",
                                            "person_pale",
                                            "person_pierced",
                                            "person_posture",
                                            "person_scarred",
                                            "person_smile",
                                            "person_surprised",
                                            "person_tattooed",
                                            "person_tired",
                                            "person_width"
                                            ], default="person_age"),
        prompt_negative: str = Input(default=""),
        seed: int = Input(default=0),
        pattern_target: str = Input(default='man'),
        delay_relative: float = Input(default=0.20),
        guidance_scale: float = Input(default=7.5),
        num_images: int = Input(default=5),
    ) -> List[Path]:
        delta = EmbeddingDelta(self.model.dims)
        state_dict = torch.load(f'/content/attribute-control/pretrained_deltas/{pretrained_deltas}.pt')
        delta.load_state_dict(state_dict['delta'])
        delta = delta.to(DEVICE)
        pattern_target = r'\b({})\b'.format(pattern_target)
        scales = np.linspace(-2, 2, num=num_images)
        characterwise_mask = get_mask_regex(prompt, pattern_target)
        emb = self.model.embed_prompt(prompt)
        emb_neg = None if prompt_negative is None else self.model.embed_prompt(prompt_negative)
        images = []
        for alpha in scales:
            img = self.model.sample_delayed(
                embs=[delta.apply(emb, characterwise_mask, alpha)],
                embs_unmodified=[emb],
                embs_neg=[emb_neg],
                delay_relative=delay_relative,
                generator=torch.manual_seed(seed),
                guidance_scale=7.5
            )[0]
            images.append(img)
        for i, img in enumerate(images):
            img.save(f'/content/{i+1}.png')
        return [Path(f'/content/{i+1}.png') for i in range(num_images)]