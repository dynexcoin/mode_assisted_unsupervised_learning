import numpy as np
from dbn2qubo import *
from exactZ import *
from v_to_h import *
def get_model_expectation_mode(rbm):
    # Returns model expectations computed with samples of the ground state of the RBM

    nodes = [rbm['w'][0].shape[0], rbm['w'][0].shape[1]]
    Q = dbn2qubo(rbm, nodes)
    fullWsize = np.prod(np.array(rbm['w'][0].shape) + 1)

    _, _, mode_v = exactZ(rbm) # <== exact for small instances; otherwise QUBO sampler required
    hidm, hid_p = v_to_h(rbm, mode_v)
    gs = np.hstack([mode_v, hidm]) # 
    model_expectation = np.dot(mode_v.T, hid_p)
    model_vis_avg = mode_v.T
    model_hidden_avg = hid_p.T

    # Ground state energy of the RBM
    eGS = -np.dot(gs, np.dot(Q, gs.T))

    # Calculate mode update learning rate (derived in the supplementary material of the paper)
    mode_push = (1/(4 * fullWsize)) * (-eGS - 0.5 * np.sum(rbm['b'][0]) - 0.5 * np.sum(rbm['b'][1]) - (1 / 4) * np.sum(rbm['w'][0]))
    #mode_push = (4*fullWsize)^(-1)*(-eGS - 0.5*sum(rbm.b{1,1}) - 0.5*sum(rbm.b{2,1}) - (1/4)*sum(rbm.W{1,1}(:)));

    return model_expectation, model_vis_avg, model_hidden_avg, mode_push

# Example usage:
# model_expectation, model_vis_avg, model_hidden_avg, mode_push = get_model_expectation_mode(rbm)
