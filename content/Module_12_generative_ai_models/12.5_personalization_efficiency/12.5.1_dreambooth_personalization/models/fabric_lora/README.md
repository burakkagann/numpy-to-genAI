---
base_model: runwayml/stable-diffusion-v1-5
library_name: peft
tags:
- lora
---
# Fabric LoRA Weights

This directory should contain the trained LoRA weights for the African fabric pattern personalization.

## Expected File

After training Exercise 3b (LoRA Training), this directory will contain:

- `pytorch_lora_weights.safetensors` - The trained LoRA adapter weights (~10-50 MB)

## How to Generate

Run the LoRA training script:

```bash
python exercise3b_train_lora.py
```

Training takes approximately 15-30 minutes on an RTX GPU.

## Pre-trained Weights

If you want to skip training, pre-trained weights may be available from the course repository.
Check the GitHub releases page for downloadable weights.

## Usage

Load these weights with:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("models/fabric_lora")
```

## Training Configuration

- Base Model: stable-diffusion-v1-5
- Instance Prompt: "a sks african fabric pattern"
- Class Prompt: "a fabric pattern"
- LoRA Rank: 4
- Learning Rate: 1e-4
- Training Steps: 800
- Prior Preservation: Enabled
### Framework versions

- PEFT 0.18.1