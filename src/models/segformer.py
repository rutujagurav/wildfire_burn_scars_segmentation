# model.py
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

def build_model(model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=2):
    # Load pretrained Segformer model
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    # Freeze pretrained encoder
    for param in model.segformer.encoder.parameters():
        param.requires_grad = False

    # Replace decode head (binary segmentation)
    model.decode_head.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    return model, processor