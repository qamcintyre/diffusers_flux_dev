Set up the environment following the README_flux.md file (get accelerate auto config + install the diffusers as an editable package). From there, just create a folder of images (jpeg) in the examples/dreambooth directory (3-5 images is ideal), cd into examples/dreambooth, and run the following command:

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="{directory of jpegs}"
export OUTPUT_DIR="trained-flux"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of {pick uncommon single token for your object}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of {pick uncommon single token for your object} on the moon" \
  --validation_epochs=25 \
  --seed="0" \
```