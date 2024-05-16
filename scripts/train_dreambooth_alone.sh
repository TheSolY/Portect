export MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./training_images/"
export CLASS_DIR="./training_images/class-person"
export OUTPUT_DIR="dreambooth-outputs/"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of ukj person" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" \
