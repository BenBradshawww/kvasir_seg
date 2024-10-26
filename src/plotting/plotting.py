import matplotlib.pyplot as plt

def plot_images(valid_images_generator, valid_masks_generator, batch_size:int):
    num_photos = min(10, batch_size)

    _, axes = plt.subplots(2, num_photos, figsize=(15, 3))

    batch_images = next(valid_images_generator)[:num_photos]
    batch_masks = next(valid_masks_generator)[:num_photos]

    for i in range(num_photos):
        axes[0, i].imshow(batch_images[i], cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(batch_masks[i], cmap='gray')
        axes[1, i].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def visualize_segmentation(input_images, true_masks, predicted_masks, batch_size:int):
    """
    Visualize the input images, true masks, and predicted masks side by side.
    
    Args:
    input_images (numpy.ndarray): The input images (batch_size, height, width, channels)
    true_masks (numpy.ndarray): The true masks (batch_size, height, width)
    predicted_masks (numpy.ndarray): The predicted masks (batch_size, height, width)
    batch_size (int): The batch size
    """

    _, axs = plt.subplots(batch_size, 3, figsize=(10, 20))

    for i in range(batch_size):
        # Display input image
        axs[i, 0].imshow(input_images[i])
        axs[i, 0].set_title('Input Image')
        axs[i, 0].axis('off')
        
        # Display true mask
        axs[i, 1].imshow(true_masks[i], cmap='gray')
        axs[i, 1].set_title('True Mask')
        axs[i, 1].axis('off')
        
        # Display predicted mask
        axs[i, 2].imshow(predicted_masks[i], cmap='gray')
        axs[i, 2].set_title('Predicted Mask')
        axs[i, 2].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    plt.show()