# hls_burn_scars_dataset.py
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio

class HLSBurnScarsDataset(Dataset):
    def __init__(self, data_dir, split='training', processor=None):
        """
        Args:
            data_dir: Path to hls_burn_scars directory
            split: 'training' or 'validation'
            processor: preprocessing function (e.g. SegformerImageProcessor)
        """
        self.data_dir = Path(data_dir) / split
        self.processor = processor
        
        # Get all merged.tif files (images)
        self.image_files = sorted([
            f for f in self.data_dir.glob("*_merged.tif")
        ])
        
        print(f"Found {len(self.image_files)} images in {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image (6 bands)
        image_path = self.image_files[idx]
        
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)  # Shape: (6, 512, 512), Values: 0 to 1``
        
        # Load mask
        mask_path = str(image_path).replace("_merged.tif", ".mask.tif")
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Shape: (512, 512), Values: 1 = Burn scar, 0 = Not burned, -1 = Missing data
        
        # Map missing (-1) to 255 (ignore_index), and ensure uint8
        mask = mask.astype(np.int32)
        mask[mask == -1] = 255
        mask = mask.astype(np.uint8)
        
        # Convert to HWC format for processor
        # Select RGB bands (bands 3,2,1 are typically RGB for HLS)
        # - 1, Blue, B02
        # - 2, Green, B03
        # - 3, Red, B04
        image = np.transpose(image[[3, 2, 1], :, :], (1, 2, 0))  # Shape: (512, 512, 3)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        # Process with SegformerImageProcessor
        if self.processor:
            self.processor.do_rescale = False  # Images are already in 0-1 range
            encoded = self.processor(
                images=image,
                segmentation_maps=mask,
                return_tensors="pt"
            )
            
            # Remove batch dimension added by processor
            pixel_values = encoded['pixel_values'].squeeze(0)
            labels = encoded['labels'].squeeze(0)
            
            return {
                'image': pixel_values,
                'mask': labels
            }
        else:
            # Return raw numpy arrays
            return {
                'image': torch.from_numpy(image).permute(2, 0, 1).float(), #/ 255.0,
                'mask': torch.from_numpy(mask).long()
            }


def create_dataloaders(data_dir, processor=None, batch_size=4, num_workers=4):
    """
    Create train, validation, and test dataloaders
    """
    train_dataset = HLSBurnScarsDataset(data_dir, 'training', processor)
    val_dataset = HLSBurnScarsDataset(data_dir, 'validation', processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get datapoint shapes
    sample_batch = next(iter(val_loader))
    input_shape = tuple(sample_batch['image'].shape)
    num_classes = len(torch.unique(sample_batch['mask'])) - (1 if 255 in torch.unique(sample_batch['mask']) else 0)
    info = {
        'input_shape': input_shape,
        'num_classes': num_classes
    }
    
    return train_loader, val_loader, info