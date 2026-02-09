# Fabric Textual Inversion Embedding

This directory should contain the trained Textual Inversion embedding for the African fabric pattern personalization.

## Expected File

After training Exercise 3a (Textual Inversion), this directory will contain:

- `learned_embeds.safetensors` - The trained token embedding (~3 KB)

## How to Generate

Run the Textual Inversion training script:

```bash
python exercise3a_train_textual_inversion.py
```

Training takes approximately 30-60 minutes.

## Usage

Load this embedding with:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_textual_inversion("models/fabric_textual_inversion/learned_embeds.safetensors")
```

Then use the placeholder token in your prompts:

```python
image = pipe("a dress made of <african-fabric> pattern").images[0]
```

## Training Configuration

- Base Model: stable-diffusion-v1-5
- Placeholder Token: "<african-fabric>"
- Initializer Token: "pattern"
- Learning Rate: 5e-4
- Training Steps: 3000
