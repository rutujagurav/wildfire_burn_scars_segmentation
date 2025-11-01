# model.py
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from peft import LoraConfig, get_peft_model, TaskType

def build_model(model_name="nvidia/segformer-b0-finetuned-ade-512-512", 
                num_classes=2):
    # Load pretrained Segformer model
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    # Freeze pretrained encoder
    for param in model.segformer.encoder.parameters():
        param.requires_grad = False

    # Replace decode head to match num_classes
    model.decode_head.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    return model, processor

def build_peft_model(base_model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value", "key"],
        lora_dropout=lora_dropout
    )

    # Wrap base model with LoRA
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model