import torch
from diffusers import FluxPipeline

# Load the base model
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the pipeline to GPU if available
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# If you used LoRA, load the LoRA weights
lora_model_path = "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-quinn"
pipe.load_lora_weights(lora_model_path)

# If you didn't use LoRA and fine-tuned the full model, you would instead load it like this:
# pipe = FluxPipeline.from_pretrained("path/to/your/trained-flux", torch_dtype=torch.float16)

# Set up the prompt
prompt = "A photo of gjg quinn"

# Generate the image
image = pipe(prompt=prompt, guidance_scale=7.5).images[0]

# Save the image
image.save("generated_image.png")