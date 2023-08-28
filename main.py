import numpy as np
import copy
import matplotlib.pyplot as plt
from create_shifting_bars import * 
from exactZ import *
from randomDBM import *
from get_data_expectation import *
from get_model_expectation_CD import *
from get_model_expectation_mode import *
from get_model_expectation_CD import *

# Clear workspace, set random seed
#np.random.seed(42)

# Define the size of the RBM
nodes = [6, 6]

# Generate some example data (replace with your actual data)
data, p_data = create_shifting_bars(nodes[0], nodes[0]//2)
print(data)

# Size of RBM ensemble
numEnsemble = 12
# Number of CD iterations
CDiters = 1
# Total parameter updates
iterations = 100000
# Plot KL every Nth update
plotPer = 1000
# CD learning rate
lr0 = 0.2
# Parameters for sigmoidal mode probability
alpha = 20
beta = -6

kls_cd_e = np.zeros((numEnsemble, iterations//plotPer))
kls_mode_e = np.zeros((numEnsemble, iterations//plotPer))

mode_probs = np.zeros((numEnsemble, 2**nodes[0]))
cd_probs = np.zeros((numEnsemble, 2**nodes[0]))

for j in range(numEnsemble):
    # Initialize RBM randomly
    print('numEnsemble =', j)
    rbm_cd = randomDBM(nodes)
    rbm_mode = copy.deepcopy(rbm_cd)

    kls_cd = np.zeros((1,iterations//plotPer))
    kls_mode = np.zeros((1,iterations//plotPer))

    for i in range(iterations-1):
        lr = lr0
        # Sample (with replacement) from training set
        datasamp = data[np.random.choice(data.shape[0], size=data.shape[0], replace=False)]
        if i == 0 or (i + 1) % plotPer == 0:
            if i == 0:
                count = 0
            else:
                count = int((i+1)/plotPer)   

            #print(j, i, count)
            # Compute exact KL divergences of two RBMs

            log_p_cd, _, _ = exactZ(rbm_cd)
            log_p_mode, _, _ = exactZ(rbm_mode)

            kl_cd = p_data * (np.log(p_data) - log_p_cd)
            kl_cd = kl_cd[~np.isnan(kl_cd)]
            kls_cd[0,count] = np.sum(kl_cd)

            kl_mode = p_data * (np.log(p_data) - log_p_mode)
            kl_mode = kl_mode[~np.isnan(kl_mode)]
            kls_mode[0,count] = np.sum(kl_mode)

        # Calculate CD update + update parameters

        cd_vhd, cd_vd, cd_hd = get_data_expectation(rbm_cd, datasamp)
        cd_vhm, cd_vm, cd_hm, _ = get_model_expectation_CD(rbm_cd, datasamp, CDiters)

        dW = lr * (cd_vhd - cd_vhm)
        dV = lr * (cd_vd - cd_vm)
        dH = lr * (cd_hd - cd_hm)

        rbm_cd['w'][0] += dW
        rbm_cd['b'][0] += dV
        rbm_cd['b'][1] += dH

        # Calculate mode training update + update parameters
        mode_vhd, mode_vd, mode_hd = get_data_expectation(rbm_mode, datasamp)

        # Compute mode sample probability
        sigm = 0.1 / (1 + np.exp(-alpha * (i+1) / iterations - beta))

        # Perform mode or CD update  np.random.rand()
        if np.random.rand() <= sigm:
            mode_vhm, mode_vm, mode_hm, mode_push = get_model_expectation_mode(rbm_mode)
            lr = lr * mode_push
        else:
            mode_vhm, mode_vm, mode_hm,_ = get_model_expectation_CD(rbm_mode, datasamp, CDiters)
            lr = lr0

        dW = lr * (mode_vhd - mode_vhm)
        dV = lr * (mode_vd - mode_vm)
        dH = lr * (mode_hd - mode_hm)

        rbm_mode['w'][0] += dW
        rbm_mode['b'][0] += dV
        rbm_mode['b'][1] += dH

    # Save KL divergences and probabilities for plotting.
    kls_cd_e[j, :] = kls_cd
    kls_mode_e[j, :] = kls_mode

    log_p_cd, _, _ = exactZ(rbm_cd)
    log_p_mode, _, _ = exactZ(rbm_mode)

    cd_probs[j, :] = (np.log(p_data) - log_p_cd).T
    mode_probs[j, :] = (np.log(p_data) - log_p_mode).T


print('training end')

# Produce plots
plotXaxis = np.arange(0, iterations, plotPer)
x = plotXaxis
xx = x
y11 = np.min(kls_cd_e, axis=0)
y12 = np.max(kls_cd_e, axis=0)
y21 = np.min(kls_mode_e, axis=0)
y22 = np.max(kls_mode_e, axis=0)
alpha = 0.5
beta = 0.1

# Define colors
modeColor = [0.8500, 0.3250, 0.0980]
cdColor = [0, 0.4470, 0.7410]

# Create new figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot for subplot (2,2,1)
aa = np.median(kls_cd_e.T, axis=1)
axs[0, 0].plot(plotXaxis, aa, linewidth=2.5, color=cdColor)
axs[0, 0].fill_between(xx, y11,y12, color=cdColor, alpha=0.2)
axs[0, 0].set_title('CD')
axs[0, 0].set_xlabel('Iterations', fontweight='bold')
axs[0, 0].set_ylabel('KL Divergence', fontweight='bold')
axs[0, 0].grid(True, which='both', linestyle=':')
axs[0, 0].set_ylim([0, 2])
axs[0, 0].set_xlim([0, iterations])


# Plot for subplot (2,2,2)
bb = np.median(kls_mode_e.T, axis=1)
axs[0, 1].plot(plotXaxis, bb, '-', linewidth=2.5, color=modeColor)
axs[0, 1].fill_between(xx, y21,y22, color=modeColor, alpha=0.2)
axs[0, 1].set_xlabel('Iterations', fontweight='bold')
axs[0, 1].set_ylabel('KL Divergence', fontweight='bold', color='black')
axs[0, 1].set_title('Mode Training')
axs[0, 1].set_ylim([0, 2])
axs[0, 1].grid(True, which='both', linestyle=':')
cc = 0.1 / (1 + np.exp(-alpha * plotXaxis / iterations - beta))
axs[0, 1].plot(plotXaxis, cc, '--', linewidth=1.0, color=[0.8500, 0.3250, 0.0980, 0.75])
axs[0, 1].twinx().set_ylabel(r'$P_{\rm{mode}}$', color='black')

# Plot for subplot (2,2,3)
axs[1, 0].bar(range(cd_probs.shape[1]), np.median(cd_probs.T, axis=1), color=cdColor)
axs[1, 0].set_xlabel('Visible Configuration (Index)', fontweight='bold')
axs[1, 0].set_ylabel('Log Probability', fontweight='bold')
axs[1, 0].grid(True, which='both', linestyle=':')
axs[1, 0].legend([r'$\log\left(\frac{q_i}{p_i}\right)$'], loc='upper left')
axs[1, 0].set_ylim([-1, 1])

# Plot for subplot (2,2,4)
axs[1, 1].bar(range(mode_probs.shape[1]) , np.median(mode_probs.T, axis=1), color=modeColor)
axs[1, 1].set_xlabel('Visible Configuration (Index)', fontweight='bold')
axs[1, 1].set_ylabel('Log Probability', fontweight='bold')
axs[1, 1].grid(True, which='both', linestyle=':')
axs[1, 1].legend([r'$\log\left(\frac{q_i}{p_i}\right)$'], loc='upper left')
axs[1, 1].set_ylim([-1, 1])



# Fine-tune the layout
plt.tight_layout()

# Save the figure or show it
plt.show()
