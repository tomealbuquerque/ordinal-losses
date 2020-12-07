import numpy as np
from copy import deepcopy
from sklearn.metrics import auc, confusion_matrix

def imbalanced_ordinal_classification_index(conf_mat, beta=None, missing='zeros', verbose=False):
    # missing: 'zeros', 'uniform', 'diagonal'
   
    N = int(np.sum(conf_mat))
    K = float(conf_mat.shape[0])
    gamma = 1.0
    if beta is None:
        beta_vals = np.linspace(0.0, 1.0, 1000).transpose()
    else:
        beta_vals = [beta]
           
    # Fixing missing classes
    conf_mat_fixed = deepcopy(conf_mat)
    for ii in range(conf_mat.shape[0]):
        if np.sum(conf_mat[ii,:]) == 0:
            if missing == 'zeros':
                K -= 1.0  # Dealt with by 0**Nr[rr]
            elif missing == 'uniform':
                conf_mat_fixed[ii,:] = np.ones((1,conf_mat.shape[1]))
            elif missing == 'diagonal':
                conf_mat_fixed[ii,ii] = 1
            else:
                raise ValueError('Unknown way of dealing with missing classes.')
           
    # Computing number of samples in each class
    Nr = np.sum(conf_mat_fixed, axis=1)
   
    beta_oc = list()
   
    # Computing total dispersion and helper matrices
    helper_mat2 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat2[rr, cc] = (float(conf_mat_fixed[rr, cc])/(Nr[rr] + 0**Nr[rr]) * ((abs(rr-cc))**gamma))
    total_dispersion = np.sum(helper_mat2)**(1/gamma)
    helper_mat1 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat1[rr, cc] = float(conf_mat_fixed[rr, cc])/(Nr[rr] + 0**Nr[rr])
    helper_mat1 = np.divide(helper_mat1, total_dispersion + K)
   
    for beta in beta_vals:
       
        beta = beta/K
       
        # Creating error matrix and filling first entry
        error_mat = np.zeros(conf_mat_fixed.shape)
        error_mat[0, 0] = 1 - helper_mat1[0, 0] + beta*helper_mat2[0, 0]

        # Filling column 0
        for rr in range(1, conf_mat_fixed.shape[0]):
            cc = 0
            error_mat[rr, cc] = error_mat[rr-1, cc] - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]

        # Filling row 0
        for cc in range(1, conf_mat_fixed.shape[1]):
            rr = 0
            error_mat[rr, cc] = error_mat[rr, cc-1] - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]

        # Filling the rest of the error matrix
        for cc in range(1, conf_mat_fixed.shape[1]):
            for rr in range(1, conf_mat_fixed.shape[0]):
                cost_up = error_mat[rr-1, cc]
                cost_left = error_mat[rr, cc-1]
                cost_lefttop = error_mat[rr-1, cc-1]
                aux = np.min([cost_up, cost_left, cost_lefttop])
                error_mat[rr, cc] = aux - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]
       
        beta_oc.append(error_mat[-1, -1])
   
    if len(beta_vals) < 2:
        return beta_oc[0]
    else:
        if verbose:
            plot_uoc(beta_vals, beta_oc)
        return auc(beta_vals, beta_oc)

def wilson_index(Y, Yhat):
    return imbalanced_ordinal_classification_index(confusion_matrix(Y, Yhat))
