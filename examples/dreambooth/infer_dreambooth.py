import torch
from diffusers import FluxPipelineMultipleLora

# Load the base model
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipelineMultipleLora.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the pipeline to GPU if available
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load multiple LoRA weights
lora_paths = [
    "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-lora",
    "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-quinn"
]

# Load multiple LoRAs
pipe.load_multiple_loras(lora_paths)

# Set up the prompt using both dreambooth tokens
prompt = "A photo of gjg quinn on the moon with sks dog"

# Generate the image with multiple LoRA scales
lora_scales = [0.7, 0.5]  # Adjust these scales as needed
image = pipe(prompt=prompt, guidance_scale=7.5).images[0]

# Save the image
image.save("generated_image_with_multiple_loras.png")
