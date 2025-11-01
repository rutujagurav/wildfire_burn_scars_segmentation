import matplotlib.pyplot as plt

def plot_datapoint(image, mask, titlestr="", show=False, save=False, save_path=None):
    """
    Plots a single datapoint with its corresponding mask.
    
    Args:
        image (np.ndarray): The input image tensor of shape (H, W, 3).
        mask (np.ndarray): The segmentation mask tensor of shape (H, W).
        titlestr (str): Title for the plot.

    Returns:
        None    
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")
    ax[1].axis('off')
    
    plt.suptitle(titlestr)
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_datapoint_with_prediction(image, mask, prediction, titlestr="", show=False, save=False, save_path=None):
    """
    Plots a single datapoint with its corresponding mask and model prediction.
    
    Args:
        image (np.ndarray): The input image tensor of shape (H, W, 3).
        mask (np.ndarray): The ground truth segmentation mask tensor of shape (H, W).
        prediction (np.ndarray): The predicted segmentation mask tensor of shape (H, W).
        titlestr (str): Title for the plot.

    Returns:
        None    
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')

    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')
    
    plt.suptitle(titlestr, y=1.05)
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()