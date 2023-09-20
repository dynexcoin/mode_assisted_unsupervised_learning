# Import libraries
import numpy as np
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Import custom modules
from create_shifting_bars import * 
from exactZ import *
from randomDBM import *
from get_data_expectation import *
from get_model_expectation_CD import *
from get_model_expectation_mode import *

# CONSTANTS
SEED = 42
NODES = [6, 6]  # [Visible nodes, Hidden nodes]
NUM_ENSEMBLES = 10  # Number of RBM ensembles
CD_ITERS = 1  # Number of Contrastive Divergence iterations
TOTAL_ITERATIONS = 20000  # Total parameter updates
PLOT_INTERVAL = 200  # Plot KL divergence every Nth update
INITIAL_LR = 0.2  # Initial learning rate for Contrastive Divergence
ALPHA = 20  # Alpha for sigmoidal mode probability
BETA = -6  # Beta for sigmoidal mode probability
P_MAX = 0.1  # Maximum mode probability


# np.random.seed(SEED) # Set the random seed
rbm_nodes = NODES # Define the size of the Restricted Boltzmann Machine (RBM)
ensemble_count = NUM_ENSEMBLES # Number of ensembles in the RBM
cd_iterations = CD_ITERS # Number of Contrastive Divergence iterations
total_iterations = TOTAL_ITERATIONS # Total number of parameter updates during training
plot_interval = PLOT_INTERVAL # Interval for plotting Kullback-Leibler divergence
initial_learning_rate = INITIAL_LR # Initial learning rate for Contrastive Divergence algorithm

# Parameters for calculating the sigmoidal mode probability
sigmoid_alpha = ALPHA
sigmoid_beta = BETA
max_mode_probability = P_MAX

# Parameters for Q_model_type and Q_solver_type
Q_model_type = 'rbm' # Choose either 'rbm' or 'dbn'
Q_solver_type = 'Sampler' # Choose either 'Exact' or 'Sampler'

def calculate_exact_kl(p_data, rbm):
    """
    Calculate the exact Kullback-Leibler (KL) divergence between the data distribution and RBM's model distribution.
    Args:
    - p_data (array): Data probability distribution
    - rbm (dict): RBM model
    
    Returns:
    - float: The calculated KL divergence
    """
    # Compute the partition function and probabilities for the model
    log_p, _, _ = exactZ(rbm)
    kl = p_data * (np.log(p_data) - log_p)
    kl = kl[~np.isnan(kl)]
    return np.sum(kl)

def update_parameters(rbm, data_sample, expectation, lr):
    """
    Update the parameters of the RBM based on data and model expectations.
    Args:
    - rbm (dict): The RBM model
    - data_sample (array): The data sample
    - expectation (array): Model expectation values
    - lr (float): Learning rate
    
    Returns:
    - dict: Updated RBM model
    """
    # Compute expectations from data and model
    vhd, vd, hd = get_data_expectation(rbm, data_sample)
    vhm, vm, hm, push_factor = expectation

    # modify learning rate based on mode push_factor
    lr = lr * push_factor if push_factor else lr

    # Perform parameter updates
    dW = lr * (vhd - vhm)
    dV = lr * (vd - vm)
    dH = lr * (hd - hm)
    rbm['w'][0] += dW
    rbm['b'][0] += dV
    rbm['b'][1] += dH

    return rbm
    
def plot_results(plotXaxis, kl_cd_ensemble, kl_mode_ensemble, cd_probs, mode_probs, sigm_values):
    """
    Plot the training and evaluation results.
    """
    # colors = sns.color_palette(["#3070b7", "#c95c2e"]) #Original colors from the paper
    colors = sns.color_palette("deep")
    cdColor = colors[0]
    modeColor = colors[1]
    
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12)) 

    # Subplot 1: CD training
    plot_single_subplot(axs[0, 0], plotXaxis, kl_cd_ensemble, 'CD', 'Iterations', 'KL Divergence', cdColor, logscale=False)

    # Subplot 2: Mode training
    plot_single_subplot(axs[0, 1], plotXaxis, kl_mode_ensemble, 'Mode Training', 'Iterations', 'KL Divergence', modeColor, logscale=False)

    # Add sigmoid curve to subplot 2
    ax2 = axs[0, 1].twinx()
    ax2.plot(plotXaxis, sigm_values.flatten(), ':', linewidth=1.0, color=modeColor)
    ax2.set_ylim([np.min(sigm_values), np.max(sigm_values)])
    ax2.set_ylabel(r'$P_{\rm{mode}}$', fontsize=14, color='black')
    ax2.grid(False)  # Disable the grid for the secondary y-axis

    # Subplot 3: CD training (log scale)
    plot_single_subplot(axs[1, 0], plotXaxis, kl_cd_ensemble, 'CD (Log scale)', 'Iterations', 'KL Divergence', cdColor, logscale=True)

    # Subplot 4: Mode training (log scale)
    plot_single_subplot(axs[1, 1], plotXaxis, kl_mode_ensemble, 'Mode Training (Log scale)', 'Iterations', 'KL Divergence', modeColor, logscale=True)

    # Subplot 5: Final probabilities for CD training
    plot_prob_subplot(axs[2, 0], cd_probs, 'Visible Configuration (Index)', 'Log Probability', cdColor)

    # Subplot 6: Final probabilities for Mode training
    plot_prob_subplot(axs[2, 1], mode_probs, 'Visible Configuration (Index)', 'Log Probability', modeColor)

    plt.tight_layout()
    plt.show()
    
def plot_single_subplot(ax, x, ensemble, title, xlabel, ylabel, color, logscale):
    """Utility function to plot a single subplot."""
    y = np.median(ensemble.T, axis=1)
    y_min, y_max = np.min(ensemble, axis=0), np.max(ensemble, axis=0)
    
    ax.plot(x, y, linewidth=2.5, color=color)
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.2)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.set_ylim([0.01, 2] if logscale else [0, 2])
    ax.set_xlim([0, total_iterations])
    if logscale:
        ax.set_yscale('log')

def plot_prob_subplot(ax, probs, xlabel, ylabel, color):
    """Utility function to plot final probabilities."""
    ax.bar(range(probs.shape[1]), np.median(probs, axis=0), color=color)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.legend([r'$\log\left(\frac{q_i}{p_i}\right)$'], fontsize=12, loc='upper left')
    ax.set_ylim([-1, 1])

def main():
    start_time = time.time()
    
    # Generate example data with shifting bars, note that you can replace this with your actual data
    data, p_data = create_shifting_bars(rbm_nodes[0], rbm_nodes[0]//2)
    
    # Initialize arrays to store KL divergence values for multiple ensembles
    kl_cd_ensemble = np.zeros((ensemble_count, total_iterations//plot_interval))
    kl_mode_ensemble = np.zeros((ensemble_count, total_iterations//plot_interval))
    
    # Initialize arrays to store probabilities calculated from different training approaches (CD and Mode)
    cd_probs = np.zeros((ensemble_count, 2**rbm_nodes[0]))
    mode_probs = np.zeros((ensemble_count, 2**rbm_nodes[0]))

    # Loop over multiple ensembles
    for ensemble_idx in range(ensemble_count):
        print()
        
        # Initialize RBM randomly for both training approaches (CD and Mode)
        rbm_cd = randomDBM(rbm_nodes)
        rbm_mode = copy.deepcopy(rbm_cd)

        # Initialize arrays to store KL divergence and sigmoid values for plotting
        kl_cd_values = np.zeros((1, total_iterations//plot_interval))
        kl_mode_values = np.zeros((1, total_iterations//plot_interval))
        sigm_values  = np.zeros((1, total_iterations//plot_interval))

        # Main training loop
        for iter_idx in tqdm(range(total_iterations), desc=f"Ensemble iteration {ensemble_idx+1}/{ensemble_count} Training Progress", ncols=100):
            # Set the learning rate, can implement learning rate decay here
            learning_rate = initial_learning_rate
            
            # Choose a random subset of data for this iteration
            data_sample = data[np.random.choice(data.shape[0], size=data.shape[0], replace=False)]

            # Compute and store KL divergence at specific intervals for plotting
            if iter_idx == 0 or (iter_idx) % plot_interval == 0:
                count = (iter_idx) // plot_interval if iter_idx != 0 else 0

                kl_cd_values[0, count] = calculate_exact_kl(p_data, rbm_cd)
                kl_mode_values[0, count] = calculate_exact_kl(p_data, rbm_mode)
            
            # Update rbm_cd using Contrastive Divergence (CD)
            cd_expectation = get_model_expectation_CD(rbm_cd, data_sample, cd_iterations)
            rbm_cd = update_parameters(rbm_cd, data_sample, cd_expectation, learning_rate)

            # Compute mode sample probability based on sigmoid function
            sigm = max_mode_probability / (1 + np.exp(-sigmoid_alpha * (iter_idx) / total_iterations - sigmoid_beta))
            sigm_values[0, count] = sigm
            
            # Update rbm_mode using either Contrastive Divergence (CD) or "Mode" based on the probability
            mode_expectation = get_model_expectation_mode(rbm_mode, Q_model_type, Q_solver_type) if np.random.rand() <= sigm else get_model_expectation_CD(rbm_mode, data_sample, cd_iterations)
            rbm_mode = update_parameters(rbm_mode, data_sample, mode_expectation, learning_rate)
        
        # Store the computed KL values for each ensemble
        kl_cd_ensemble[ensemble_idx, :] = kl_cd_values
        kl_mode_ensemble[ensemble_idx, :] = kl_mode_values

        # Compute the final probabilities for each model
        log_p_cd, _, _ = exactZ(rbm_cd)
        log_p_mode, _, _ = exactZ(rbm_mode)
        
        # Store the final probabilities for each ensemble and method
        cd_probs[ensemble_idx, :] = (np.log(p_data) - log_p_cd).T
        mode_probs[ensemble_idx, :] = (np.log(p_data) - log_p_mode).T
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training End, took {elapsed_time:.2f} secs')
    plotXaxis = np.arange(0, total_iterations, plot_interval)
    plot_results(plotXaxis, kl_cd_ensemble, kl_mode_ensemble, cd_probs, mode_probs, sigm_values)

if __name__ == "__main__":
    main()
