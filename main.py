import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from marbm import MARBM

def plot_free_energies(rbm_cd, rbm_mode, mode = 'free_energy'):
    plt.figure(figsize=(12, 6))
    plt.plot(rbm_cd.metrics, label="CD Trained RBM")
    plt.plot(rbm_mode.metrics, label="Mode Assisted RBM")
    if mode == 'free_energy':
        plt.title('Free Energy over Training Steps')
    else:
        plt.title('KL loss over Training Steps')
    plt.xlabel('Steps')
    if mode == 'free_energy':
        plt.ylabel('Free Energy')
    else:
        plt.ylabel('KL loss')
    plt.legend()
    plt.grid(True)
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
    hidden_units = 64
    epochs = 12
    lr = 0.1
    k = 1
    sigm_a=20
    sigm_b=-6
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

    rbm_cd = MARBM(visible_units, hidden_units)
    rbm_cd.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=0.0, plotper=plotper, loss_metric='free_energy')
    
    rbm_mode = MARBM(visible_units, hidden_units)
    rbm_mode.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=0.1, plotper=plotper, loss_metric='free_energy')

    # Using validation data for visualization
    sample_val_data = next(iter(val_loader))[0]
    reconstructed_data_cd = rbm_cd.reconstruct(sample_val_data)
    reconstructed_data_mode = rbm_mode.reconstruct(sample_val_data)

    display_reconstructions(sample_val_data, reconstructed_data_cd, reconstructed_data_mode)
    
    # Compute and print free energies
    free_energy_cd = rbm_cd.compute_free_energy(sample_val_data)
    free_energy_mode = rbm_mode.compute_free_energy(sample_val_data)
    print(f"Free Energy (CD Trained RBM): {free_energy_cd}")
    print(f"Free Energy (Mode Assisted RBM): {free_energy_mode}")
    
    plot_free_energies(rbm_cd, rbm_mode, 'free_energy')

if __name__ == "__main__":
    main()