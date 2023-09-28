import numpy as np
import dimod
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(filename="MARBM.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class MARBM(nn.Module):
    """
    Mode-Assisted Restricted Boltzmann Machine (MARBM).
    
    Restricted Boltzmann machines (RBMs) are a class of generative models that
    have historically been challenging to train due to the complex nature of 
    their gradient. The MARBM provides a novel approach to training RBMs by 
    combining the standard gradient updates with an off-gradient direction. 
    This direction is constructed using samples from the RBM ground state, 
    also referred to as 'mode'. By utilizing mode-assisted training, the RBM 
    benefits from faster convergence, improved training stability, and lower 
    converged relative entropy (KL divergence).
    
    
    Parameters:
    - visible_units (int): Number of visible units in the RBM.
    - hidden_units (int): Number of hidden units in the RBM.
    
    Attributes:
    - W (torch.Tensor): Weights connecting the visible and hidden units.
    - h_bias (torch.Tensor): Biases associated with the hidden units.
    - v_bias (torch.Tensor): Biases associated with the visible units.
    - free_energies (list): List to store computed free energies during training.
    
    Methods:
    - forward: Compute the forward pass (probability of hidden given visible).
    - sample_hidden: Sample from the hidden layer given the visible layer.
    - sample_visible: Sample from the visible layer given the hidden layer.
    - contrastive_divergence: Perform a Contrastive Divergence (CD) step.
    - rbm2qubo: Convert RBM parameters to a QUBO matrix.
    - train: Train the RBM using mode-assisted training.
    - reconstruct: Reconstruct input data using the trained RBM.
    - _compute_free_energy: Compute the free energy of a given configuration.
    - _mode_train_step: Execute one step of mode-assisted training.
    - _cd_train_step: Execute one step of training using Contrastive Divergence.
    
    """
    def __init__(self, visible_units: int, hidden_units: int):
        """
        Initializes the Mode-Assisted Restricted Boltzmann Machine (RBM).
        
        Parameters:
        ----------
        visible_units : int
            The number of visible units. Must be a positive integer.
        hidden_units : int
            The number of hidden units. Must be a positive integer.
            
        Attributes:
        -----------
        visible_units : int
            The number of visible units in the RBM.
        hidden_units : int
            The number of hidden units in the RBM.
        W : torch.nn.Parameter
            Weight matrix initialized with random values.
        h_bias : torch.nn.Parameter
            Bias vector for hidden units, initialized to zeros.
        v_bias : torch.nn.Parameter
            Bias vector for visible units, initialized to zeros.
        free_energies : list
            A list to store the free energies during training.
        """
        # Call the constructor of the parent class (nn.Module)
        super(MARBM, self).__init__()

        # Check the validity of the input parameters
        assert isinstance(visible_units, int) and visible_units > 0, "visible_units should be a positive integer."
        assert isinstance(hidden_units, int) and hidden_units > 0, "hidden_units should be a positive integer."
        
        # Initialization of the number of visible and hidden units
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        # Initialization of weights and biases
        # Note: Weights are initialized with small random values to break the symmetry during training
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        
        # Note: Biases are initialized to zeros as a common practice in RBM training
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        
        # Initialize the list to store free energies
        self.free_energies = []

        # Log the initialization details
        logger.info("Initialized MARBM with visible units: %s, hidden units: %s", visible_units, hidden_units)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.
        
        Given the visible units, compute the probability of the hidden units being activated.
        
        Parameters:
            v (torch.Tensor): The visible units.
        
        Returns:
            torch.Tensor: Probability of hidden units being activated.
        """
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        logger.debug("Completed forward pass")
        return h_prob

    def sample_hidden(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample from the hidden layer given the visible layer.

        Given the state of the visible units, this method computes the probability 
        of each hidden unit being activated and then samples from a Bernoulli 
        distribution based on these probabilities.

        Parameters:
        - v (torch.Tensor): A tensor representing the state of the visible units. It should have a shape of (batch_size, visible_units).

        Returns:
        - torch.Tensor: A tensor representing the sampled state of the hidden units. It will have a shape of (batch_size, hidden_units).
        """
        h_prob = self.forward(v)
        h_sample = torch.bernoulli(h_prob)
        logger.debug("Sampled hidden layer with shape: %s", str(h_sample.shape))
        return h_sample

    def sample_visible(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample from the visible layer given the hidden layer.

        Given the state of the hidden units, this method computes the probability 
        of each visible unit being activated and then samples from a Bernoulli 
        distribution based on these probabilities.

        Parameters:
        - h (torch.Tensor): A tensor representing the state of the hidden units. It should have a shape of (batch_size, hidden_units).

        Returns:
        - torch.Tensor: A tensor representing the sampled state of the visible units. It will have a shape of (batch_size, visible_units).
        """
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        v_sample = torch.bernoulli(v_prob)
        logger.debug("Sampled visible layer with shape: %s", str(v_sample.shape))
        return v_sample

    def contrastive_divergence(self, input_data: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Perform one step of Contrastive Divergence (CD) for training the RBM.

        The method approximates the gradient of the log-likelihood of the data 
        by running a Gibbs chain for a specified number of steps, k.
        
        Parameters:
        - input_data (torch.Tensor): The visible layer data, of shape [batch_size, visible_units].
        - k (int, optional): The number of Gibbs sampling steps. Defaults to 1.

        Returns:
        - torch.Tensor: The difference between the outer product of the data and 
                        hidden probabilities at the start and the end of the Gibbs chain, 
                        of shape [visible_units, hidden_units].
        """
        
        v0 = input_data
        vk = v0.clone()  # Use clone to avoid in-place modification issues
        
        for _ in range(k):
            hk = self.sample_hidden(vk)
            vk = self.sample_visible(hk)

        h0_prob = self.forward(v0)
        hk_prob = self.forward(vk)

        positive_phase = torch.mm(v0.t(), h0_prob)
        negative_phase = torch.mm(vk.t(), hk_prob)

        logger.debug("Performed Contrastive Divergence")
        return positive_phase - negative_phase
        
    def rbm2qubo(self) -> np.ndarray:
        """
        Convert RBM parameters to a QUBO (Quadratic Unconstrained Binary Optimization) matrix.
        
        The QUBO matrix is constructed using the weights and biases of the RBM. The diagonal 
        of the QUBO matrix corresponds to biases, and the off-diagonal elements correspond to the weights.

        Returns:
            numpy.ndarray: The QUBO matrix with shape (n_total, n_total), where n_total = n_visible + n_hidden.
        """
        
        # Extract the parameters from the RBM as numpy arrays
        vishid = self.W.detach().numpy()
        hidbiases = self.h_bias.detach().numpy().flatten()
        visbiases = self.v_bias.detach().numpy().flatten()
        
        # Number of visible and hidden nodes
        n_visible, n_hidden = vishid.shape
        n_total = n_visible + n_hidden
        
        # Initialize the QUBO matrix with zeros
        Q = np.zeros((n_total, n_total))
        
        # Populate the diagonal entries with biases
        Q[:n_visible, :n_visible] = np.diag(hidbiases)
        Q[n_visible:, n_visible:] = np.diag(visbiases)
        
        # Populate the off-diagonal entries with weights
        Q[:n_visible, n_visible:] = vishid
        return Q
        
    def train(self, train_loader, epochs=10, lr=0.01, k=1, sigm_a=20, sigm_b=-6, p_max=0.1, plotper=100):
        """
        Trains the MARBM model using given data.

        Parameters:
        - train_loader (torch.utils.data.DataLoader): DataLoader containing the training data and labels.
        - epochs (int, optional): Number of training epochs. Default is 10.
        - lr (float, optional): Learning rate for the optimizer. Default is 0.01.
        - k (int, optional): Number of Gibbs sampling steps for contrastive divergence. Default is 1.
        - sigm_a (float, optional): Parameter for the sigmoidal function determining mode switching. Default is 20.
        - sigm_b (float, optional): Parameter for the sigmoidal function determining mode switching. Default is -6.
        - p_max (float, optional): Maximum probability for the sigmoidal switch function. Should be in the range (0, 1]. Default is 0.1.
        - plotper (int, optional): Interval at which free energy is computed and logged. Default is 100.

        Notes:
        The training process alternates between mode training and contrastive divergence based on a stochastic switch.
        The free energy of the model is computed and stored at intervals defined by the `plotper` argument.
        """

        assert lr > 0, "Learning rate (lr) should be positive."
        assert sigm_a > 0, "sigm_a should be positive."
        assert 0 <= p_max <= 1, "p_max should be in the range (0, 1]."
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs
        logger.info("Total training steps: %s", total_steps)

        logger.info("Training started for %s epochs", epochs)
        
        for epoch in range(epochs):
            logger.info("Epoch %s started", epoch+1)
            for iter_idx, (data, _) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training Progress", ncols=100, total=len(train_loader)):
                
                sigm = p_max / (1 + np.exp(-sigm_a * (iter_idx + epoch * steps_per_epoch) / total_steps - sigm_b))
                if torch.rand(1) <= sigm:
                    self._mode_train_step(data, optimizer, lr)
                else:
                    self._cd_train_step(data, optimizer, lr, k)
                
                # Calculate free energy every 'plotper' steps
                if iter_idx % plotper == 0:
                    with torch.no_grad():
                        free_energy = self._compute_free_energy(data).mean().item()
                        self.free_energies.append(free_energy)

        logger.info("Training completed")
        
    def _mode_sampling(self):
        """
        Uses simulated annealing to sample the mode (ground state) of the RBM encoded as a QUBO.

        Returns:
        - mode_v (np.array): Visible units' state of the sampled mode.
        - mode_h (np.array): Hidden units' state of the sampled mode.
        - ground_state_energy (float): Energy of the sampled ground state.
        """
        Q = self.rbm2qubo()
        simulated_annealing_parameters = {
            'beta_range': [0.1, 1.0],
            'num_reads': 2,
            'num_sweeps': 25
        }
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(-Q, **simulated_annealing_parameters)
        ground_state = response.first.sample
        ground_state_energy = response.first.energy

        mode_v = np.array([ground_state[i] for i in range(self.visible_units)]).reshape(1, -1)
        mode_h = np.array([ground_state[i] for i in range(self.visible_units, self.visible_units + self.hidden_units)]).reshape(1, -1)
        
        return mode_v, mode_h, ground_state_energy
    
    def _mode_train_step(self, input_data, optimizer, lr):
        """
        Performs a training step using mode-guided training for the RBM.

        This method applies a quantum-inspired simulated annealing to sample the ground state
        (mode) of the RBM encoded as a QUBO. The mode, once sampled, is used to derive 
        model-driven expectations. These expectations, combined with data-driven expectations,
        guide the optimization of the RBM's parameters.

        Parameters:
        - input_data (torch.Tensor): A batch of training data with shape (batch_size, visible_units).
        - optimizer (torch.optim.Optimizer): The optimizer instance used for parameter updates.
        - lr (float): Base learning rate for parameter updates.
        """
        optimizer.zero_grad()

        # Sampling the mode
        mode_v, mode_h, ground_state_energy = self._mode_sampling()
        
        # Calculate model expectations and averages
        model_expectation = torch.mm(torch.tensor(mode_v.T, dtype=torch.float32), torch.tensor(mode_h, dtype=torch.float32))
        model_vis_avg = torch.tensor(mode_v.T, dtype=torch.float32)
        model_hidden_avg = torch.tensor(mode_h.T, dtype=torch.float32)
        
        # Calculate data-driven expectations and averages (positive phase)
        h0_prob = self.forward(input_data)
        data_expectation = torch.mm(input_data.t(), h0_prob)
        data_vis_avg = torch.mean(input_data, dim=0)
        data_hidden_avg = torch.mean(h0_prob, dim=0)
        
        # Calculate the full weight size and mode push
        fullWsize = np.prod(np.array([self.visible_units, self.hidden_units]) + 1)
        mode_push = (1 / (4 * fullWsize)) * (-ground_state_energy - 0.5 * np.sum(self.h_bias.detach().numpy()) - 0.5 * np.sum(self.v_bias.detach().numpy()) - (1 / 4) * np.sum(self.W.detach().numpy()))
        
        optimizer.param_groups[0]['lr'] = lr * mode_push
        
        # Calculate the gradients based on differences in expectations
        self.W.grad = -(data_expectation - model_expectation).t() / input_data.shape[0]
        self.v_bias.grad = -(data_vis_avg - model_vis_avg).mean(dim=0)
        self.h_bias.grad = -(data_hidden_avg - model_hidden_avg).mean(dim=0)
        optimizer.step()
    
    def _cd_train_step(self, input_data, optimizer, lr, k=1):
        """
        Performs one step of training using Contrastive Divergence (CD).
        
        The function updates the model's parameters using 
        the provided optimizer and learning rate.

        Parameters:
        - input_data (torch.Tensor): The input data tensor of shape [batch_size, visible_units].
        - optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        - lr (float): The learning rate for the optimizer.
        - k (int, optional): The number of Gibbs sampling steps used in CD. Default is 1.
        """
        optimizer.zero_grad()
        optimizer.param_groups[0]['lr'] = lr
        weight_grad = self.contrastive_divergence(input_data, k=k)
        # Updating weights and biases using gradients from CD
        self.W.grad = -weight_grad.t() / input_data.shape[0]
        self.v_bias.grad = -(input_data - self.sample_visible(self.sample_hidden(input_data))).mean(dim=0)
        self.h_bias.grad = -(self.forward(input_data) - self.forward(self.sample_visible(self.sample_hidden(input_data)))).mean(dim=0)
        optimizer.step()

    def reconstruct(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input data by passing it through the RBM's hidden layer and then back to the visible layer.

        Given an input visible layer, this method computes the activation of the hidden layer using the method `sample_hidden`,
        and then reconstructs the visible layer using the method `sample_visible`. This is a common approach in RBMs for data reconstruction.

        Parameters:
        - input_data (torch.Tensor): A tensor representing the visible layer's data to be reconstructed. 
                                    Shape should be (batch_size, visible_units).

        Returns:
        - torch.Tensor: A tensor of the reconstructed visible layer. Shape is (batch_size, visible_units).

        Example:
        ```
        rbm = MARBM(visible_units=784, hidden_units=500)
        input_tensor = torch.rand((32, 784))
        reconstructed_tensor = rbm.reconstruct(input_tensor)
        ```
        """
        h = self.sample_hidden(input_data)
        v = self.sample_visible(h)
        logger.debug("Reconstructed input data with shape: %s", str(v.shape))
        return v

    def _compute_free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a given configuration.

        The free energy is calculated using the formula:
        F(v) = - v * v_bias - Σ softplus(W * v + h_bias)

        Parameters:
        v (torch.Tensor): The visible layer configuration, of shape [batch_size, visible_units].

        Returns:
        torch.Tensor: The computed free energy for each configuration in the batch, of shape [batch_size].
        """
        wx_b = F.linear(v, self.W, self.h_bias)
        term_1 = torch.matmul(v, self.v_bias)
        term_2 = torch.sum(F.softplus(wx_b), dim=1)
        logger.debug("Computed free energy for a configuration")
        return -term_1 - term_2
        
    def save_model(self, path):
        """
        Save the trained weights and biases of the RBM.
        
        Parameters:
            - path (str): Path to save the model's state.
        """
        torch.save({
            'W': self.W.state_dict(),
            'h_bias': self.h_bias.state_dict(),
            'v_bias': self.v_bias.state_dict()
        }, path)
        
    def load_model(self, path):
        """
        Load the weights and biases of the RBM from a saved state.
        
        Parameters:
            - path (str): Path from where to load the model's state.
        """
        checkpoint = torch.load(path)
        self.W.load_state_dict(checkpoint['W'])
        self.h_bias.load_state_dict(checkpoint['h_bias'])
        self.v_bias.load_state_dict(checkpoint['v_bias'])
