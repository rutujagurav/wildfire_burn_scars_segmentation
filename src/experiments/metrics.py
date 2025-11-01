import numpy as np
import segmentation_models_pytorch as smp

def compute_iou(preds, masks, num_classes):
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(preds == cls, masks == cls).sum()
        union = np.logical_or(preds == cls, masks == cls).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def compute_accuracy(preds, masks):
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    correct = (preds == masks).sum()
    total = masks.size
    return correct / total

def compute_metrics(preds, masks):
    """
    Computes IoU, F1-score, Accuracy, and Recall using segmentation_models_pytorch.
    
    Args:
        preds (torch.Tensor): Predicted segmentation masks of shape (N, num_classes, H, W).
        masks (torch.Tensor): Ground truth segmentation masks of shape (N, num_classes, H, W).

    Returns:
        dict: Dictionary containing IoU, F1-score, Accuracy, and Recall.
    """
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='multilabel', threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    
    return {
        "iou": iou_score,
        "f1": f1_score,
        "f2": f2_score,
        "accuracy": accuracy,
        "recall": recall
    }