import os, sys
PROJECT_NAME = "wildfire_burn_scars_segmentation"
PROJECT_DIR = os.path.expanduser(f"~/{PROJECT_NAME}")
print(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

import json, traceback, datetime, random, csv
from pprint import pprint
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchinfo import summary
# import optuna

from models.segformer import build_model, build_peft_model
from dataloading.hls_burn_scars_dataset import create_dataloaders
from experiments.metrics import compute_metrics, compute_iou, compute_accuracy

# ---------------------------
# Argument parser
# ---------------------------
import argparse
parser = argparse.ArgumentParser(description="Train SegFormer on HLS Burn Scars dataset")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default=f"{PROJECT_DIR}/data/hls_burn_scars", help='Path to hls_burn_scars directory')
parser.add_argument('--results_dir', type=str, default=f"{PROJECT_DIR}/results", help='Path to results directory')
parser.add_argument('--exp_name', type=str, default='simpletune-1', help='Experiment name')
parser.add_argument('--model_name', type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
parser.add_argument('--model_params', type=str, nargs='+', default=['use_peft=False', 'lora_r=8', 'lora_alpha=16', 'lora_dropout=0.1'], help='Model parameters as key=value pairs')
parser.add_argument('--training_params', type=str, nargs='+', default=['batch_size=4', 'epochs=100', 'lr=5e-4'], help='Training parameters as key=value pairs')
parser.add_argument('--earlystop', action='store_true', help='Whether to use early stopping based on validation loss')
parser.add_argument('--earlystop_params', type=str, nargs='+', default=['patience=10', 'min_delta=0.0001', 'warmup_epochs=5'], help='Early stopping parameters as key=value pairs')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda:0')


# ---------------------------
# Utility functions
# ---------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_params(params_list):
    """
    Parse list of key=value strings into a dictionary.
    """
    params_dict = {}
    for param in params_list:
        key, value = param.split('=')
        try: value = int(value)
        except: 
            try: value = float(value)
            except:
                if value.lower() == 'true': value = True
                elif value.lower() == 'false': value = False
        params_dict[key] = value
    return params_dict

# ---------------------------
# Training & Eval
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_iou, n = 0.0, 0.0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        
        outputs = model(pixel_values=images)
        logits = outputs.logits  # (B, num_classes, H, W)

        # Upsample logits to match mask size
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        iou = compute_iou(preds, masks, num_classes=2)
        # metrics = compute_metrics(preds, masks.round().long())

        bs = images.size(0)
        running_loss += loss.item() * bs
        if not np.isnan(iou):
            running_iou += iou * bs
        n += bs

    return {'loss': running_loss / n, 'iou': running_iou / n}


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_iou, n = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Upsample logits to match mask size
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, masks)

            preds = torch.argmax(logits, dim=1)
            iou = compute_iou(preds, masks, num_classes=2)
            # metrics = compute_metrics(preds, masks.round().long())

            bs = images.size(0)
            running_loss += loss.item() * bs
            if not np.isnan(iou):
                running_iou += iou * bs
            n += bs

    return {'loss': running_loss / n, 'iou': running_iou / n}


# ---------------------------
# Main
# ---------------------------
def main():
    
    tic = datetime.datetime.now()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Parse training params
    training_params = parse_params(args.training_params)
    model_params = parse_params(args.model_params)
    earlystop_params = parse_params(args.earlystop_params)

    logging.info(f"Using device: {device}")
    logging.info(f"Seed: {args.seed}")

    # Results directory
    RESULTS_DIR = os.path.join(args.results_dir, args.exp_name, args.model_name.replace('/', '_'))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots', 'train'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots', 'test'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'scripts'), exist_ok=True)
    logging.info(f"Results will be saved to: {RESULTS_DIR}")

    ## Save copy of this script to RESULTS_DIR along with args
    os.system(f"cp {f"{PROJECT_DIR}/src/experiments/train.py"} {os.path.join(RESULTS_DIR, 'scripts', 'train.py')}")
    os.system(f"cp {f"{PROJECT_DIR}/src/models/segformer.py"} {os.path.join(RESULTS_DIR, 'scripts', 'segformer.py')}")
    os.system(f"cp {f"{PROJECT_DIR}/src/dataloading/hls_burn_scars_dataset.py"} {os.path.join(RESULTS_DIR, 'scripts', 'hls_burn_scars_dataset.py')}")
    with open(os.path.join(RESULTS_DIR, 'scripts', 'train.json'), 'w') as f: json.dump(vars(args), f, indent=4)


    # Data
    train_loader, val_loader, data_info = create_dataloaders(
        args.data_dir, batch_size=training_params['batch_size'], num_workers=args.num_workers
    )
    input_shape = data_info['input_shape']
    num_classes = data_info['num_classes']
    logging.info(f"Input shape: {input_shape}, Num classes: {num_classes}")
    

    # Model
    if model_params['use_peft']:
        base_model, _ = build_model(num_classes=num_classes, model_name=args.model_name)
        peft_params = model_params.copy()
        peft_params.pop('use_peft')
        model = build_peft_model(base_model, **peft_params)
    else:
        model, _ = build_model(num_classes=num_classes, model_name=args.model_name)

    # for name, module in model.named_modules():
    #     if "attention" in name:
    #         print(name)

    model = model.to(device)
    model_summary_path = os.path.join(RESULTS_DIR, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        f.write(str(summary(model, data=(next(iter(train_loader))))))
    logging.info(f"Model summary saved to {model_summary_path}")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['lr'])

    best_val_loss = float('inf')
    best_model_path = os.path.join(RESULTS_DIR, "checkpoints", "best_model.pth")

    training_stats = {
        'epoch':[],
        'loss': {'train': [], 'val': []},
        'iou': {'train': [], 'val': []}
    }
    training_stats_log_path = os.path.join(RESULTS_DIR, 'plots', 'train', "training_logs.json")
    no_improve_epochs = 0
    for epoch in range(1, training_params['epochs'] + 1):
        train_logs = train_epoch(model, train_loader, criterion, optimizer, device)
        val_logs = evaluate(model, val_loader, criterion, device)

        training_stats['epoch'].append(epoch)
        training_stats['loss']['train'].append(train_logs['loss'])
        training_stats['loss']['val'].append(val_logs['loss'])
        training_stats['iou']['train'].append(train_logs['iou'])
        training_stats['iou']['val'].append(val_logs['iou'])

        # Print epoch summary
        if epoch == 1 or epoch % 10 == 0:
            logging.info(f"\nEpoch {epoch}/{training_params['epochs']} Train Loss: {train_logs['loss']:.4f}, Val Loss: {val_logs['loss']:.4f}, Train IoU: {train_logs['iou']:.4f}, Val IoU: {val_logs['iou']:.4f}")
            # Save example predictions
            with torch.no_grad():
                batch0 = next((b for b in val_loader if b is not None), None)
                images = batch0['image'].to(device)
                masks = batch0['mask'].to(device)
                outputs = model(pixel_values=images)
                logits = outputs.logits
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                preds = torch.argmax(logits, dim=1)
                # Save images, masks, preds as .npy
                sample_save_path = os.path.join(RESULTS_DIR, 'predictions', f'epoch{epoch:04d}.npz')
                np.savez_compressed(sample_save_path, images=images.cpu().numpy(), masks=masks.cpu().numpy(), preds=preds.cpu().numpy())
        
        # Log epoch stats to CSV
        csv_path = os.path.join(RESULTS_DIR, 'plots', 'train', 'running_training_stats.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = (epoch == 1 and not os.path.exists(csv_path))
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou"])
            writer.writerow([epoch, train_logs['loss'], val_logs['loss'], train_logs['iou'], val_logs['iou']])

        improved = val_logs['loss'] < best_val_loss
        if improved:
            best_val_loss = val_logs['loss']
            best_epoch = epoch
            best_model_state = model.state_dict()
            no_improve_epochs = 0

            torch.save({
                'epoch': best_epoch,
                'model_params': {'model_name': args.model_name, 'num_classes': num_classes},
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss
            }, best_model_path)
        else:
            no_improve_epochs += 1
        
        # Check early stopping condition
        if args.earlystop:
            if epoch > earlystop_params.get('warmup_epochs', 0) and no_improve_epochs >= earlystop_params.get('patience', 10):
                logging.info(f"Early stopping at epoch {epoch} after {no_improve_epochs} epochs with no improvement.")
                break

    # Save logs
    log_path = os.path.join(RESULTS_DIR, "training_logs.json")
    with open(log_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    logging.info(f"Training logs saved to {log_path}")

    logging.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logging.info(f"Elapsed: {datetime.datetime.now() - tic}")


if __name__ == "__main__":
    main()
