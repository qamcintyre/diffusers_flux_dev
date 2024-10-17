# import torch
# from diffusers import FluxPipeline

# # Load the base model
# model_id = "black-forest-labs/FLUX.1-dev"
# pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# # Move the pipeline to GPU if available
# pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# # If you used LoRA, load the LoRA weights
# lora_model_path = "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-quinn"
# pipe.load_lora_weights(lora_model_path)

# # If you didn't use LoRA and fine-tuned the full model, you would instead load it like this:
# # pipe = FluxPipeline.from_pretrained("path/to/your/trained-flux", torch_dtype=torch.float16)

# # Set up the prompt
# prompt = "A photo of gjg quinn in a conversation with taylor swift IMG_20240601_163033.jpg"

# # Generate the image
# image = pipe(prompt=prompt, guidance_scale=7.5).images[0]

# # Save the image
# image.save("generated_image.png")

import torch
from diffusers import FluxPipeline
from peft import PeftModel
import os

# Load the base model
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the pipeline to GPU if available
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define paths to the two LoRA models
lora_model_path1 = "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-quinn"
lora_model_path2 = "/home/bfs/quinn/flux/diffusers/examples/dreambooth/trained-flux-lora"

# Function to load and combine LoRA weights
def combine_lora_weights(pipe, lora_path1, lora_path2):
    # Load the first LoRA
    pipe.load_lora_weights(lora_path1)
    lora1 = pipe.text_encoder.peft_config
    state_dict1 = pipe.text_encoder.state_dict()
    
    # Load the second LoRA
    pipe.load_lora_weights(lora_path2)
    lora2 = pipe.text_encoder.peft_config
    state_dict2 = pipe.text_encoder.state_dict()
    
    # Combine weights
    combined_state_dict = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            if 'lora' in key:
                combined_state_dict[key] = state_dict1[key] + state_dict2[key]
            else:
                combined_state_dict[key] = state_dict1[key]
        else:
            combined_state_dict[key] = state_dict1[key]
    
    # Apply combined weights
    pipe.text_encoder.load_state_dict(combined_state_dict)
    
    # Combine and set LoRA configs
    combined_lora_config = lora1
    combined_lora_config.update(lora2)
    pipe.text_encoder.peft_config = combined_lora_config
    
    return pipe

# Combine and apply LoRA weights
pipe = combine_lora_weights(pipe, lora_model_path1, lora_model_path2)

# Set up the prompt
prompt = "A photo of gjg quinn in a conversation with taylor swift IMG_20240601_163033.jpg"

# Generate the image
image = pipe(prompt=prompt, guidance_scale=7.5).images[0]

# Save the image
image.save("generated_image.png")
