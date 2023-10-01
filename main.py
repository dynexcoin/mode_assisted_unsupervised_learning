import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from marbm import MARBM


def plot_visualization_data(rbm_cd, rbm_mode):
    # Get visualization data from RBMs
    metrics_name_cd, metrics_values_cd, sigm_values_cd = rbm_cd.get_visualization_data()
    metrics_name_mode, metrics_values_mode, sigm_values_mode = rbm_mode.get_visualization_data()
    
    # Ensure both RBMs are tracking the same metric
    assert metrics_name_cd == metrics_name_mode, "The two RBMs are tracking different metrics!"

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot metrics for CD Trained RBM
    ax1.plot(metrics_values_cd, label="CD Trained RBM")
    # Plot metrics for Mode Assisted RBM
    ax1.plot(metrics_values_mode, label="Mode Assisted RBM")
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(metrics_name_cd)
    ax1.tick_params('y')
    ax1.legend(loc='upper left')

    # Set up ax2 to be the second y-axis with shared x-axis
    ax2 = ax1.twinx()
    # Plot sigmoid values for CD Trained RBM
    if sigm_values_cd:
        ax2.plot(sigm_values_cd, '--', label="CD Trained RBM Sigm Values")
    # Plot sigmoid values for Mode Assisted RBM
    if sigm_values_mode:
        ax2.plot(sigm_values_mode, '--', label="Mode Assisted RBM Sigm Values")
    ax2.set_ylabel('Sigmoid Value')
    ax2.tick_params('y')
    ax2.legend(loc='upper right')

    # Set title and grid
    plt.title(f'{metrics_name_cd} and Sigmoid Values over Training Steps')
    ax1.grid(True)
    
    # Show the plot
    plt.show()

def display_reconstructions(original, reconstructed_cd, reconstructed_mode):
    num_images = 6
    fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(12, 8))

    # Define a utility function to reduce repetition
    def plot_image(ax, img, title, show_cbar=False):
        im = ax.imshow(img.detach().numpy().reshape(20, 20), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
            cbar.ax.tick_params(labelsize=8)

    for i in range(num_images):
        # Display original images
        plot_image(axes[0, i], original[i], "Original" if i == 0 else "")
        
        # Display reconstructed images (CD)
        plot_image(axes[1, i], reconstructed_cd[i], "CD" if i == 0 else "")

        # Display reconstructed images (Mode Assisted)
        plot_image(axes[2, i], reconstructed_mode[i], "Mode Assisted" if i == 0 else "")

    plt.tight_layout()
    plt.show()

def main():
    batch_size = 8
    visible_units = 20 * 20
    hidden_units = 20
    epochs = 10
    lr = 0.1
    k = 1
    sigm_a=20
    sigm_b=-6
    p_max=0.002
    plotper=1000
    
    
    transform = transforms.Compose([
        transforms.Resize((20, 20)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x.view(-1)))
    ])   
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Split the dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

    # MARBM class then train
    rbm_cd = MARBM(visible_units, hidden_units)
    rbm_cd.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=0.0, plotper=plotper, loss_metric='free_energy')
    
    rbm_mode = MARBM(visible_units, hidden_units)
    rbm_mode.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=p_max, plotper=plotper, loss_metric='free_energy')
    
    sample_val_data = next(iter(val_loader))[0]
    
    # lock_weights test
    rbm_mode.lock_weights()
    features_mode = rbm_mode.extract_features(sample_val_data)
    print("Features from rbm_mode:", features_mode.shape)
    rbm_mode.unlock_weights()  
    
    # Save and Load Test
    model_save_path = "./rbm_mode_checkpoint.pth"
    rbm_mode.save_model(model_save_path)
    rbm_mode_new = MARBM(visible_units, hidden_units)
    rbm_mode_new.load_model(model_save_path)

    # Ensure the loaded model produces the same output as the original model
    features_from_saved_model = rbm_mode_new.extract_features(sample_val_data)
    assert torch.allclose(features_mode, features_from_saved_model), "The loaded model does not match the original model!"
    print("Successfully saved and loaded the model!")
    
    # Using validation data for visualization
    reconstructed_data_cd = rbm_cd.reconstruct(sample_val_data)
    reconstructed_data_mode = rbm_mode.reconstruct(sample_val_data)
    display_reconstructions(sample_val_data, reconstructed_data_cd, reconstructed_data_mode)
    
    plot_visualization_data(rbm_cd, rbm_mode)

if __name__ == "__main__":
    main()