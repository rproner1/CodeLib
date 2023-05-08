import numpy as np
from joblib import delayed, Parallel


def mc_shapley_value_par(estimator, feature_index, instance_index, X, n_iters, n_jobs=-1, lstm=False):
    
    """
    Estimates shapley value of feature 'feature_index', for instance 'instance_index', under the estimator provided.
    
    Parameters
    ----------
    estimator: a trained sklearn model or keras single- or multi-task neural network.
    feature_index: a feature index in [0, n_features-1].
    instance_index: the row index of the instance of interest in df set X. 
    X: dataframe of feature values.
    n_iters: the number of monte carlo iterations
    n_jobs: number of cores to use for creating feature permutations in parallel (default=-1, i.e., all available cores).
    
    Returns
    ----------
    shapley_value: a monte carlo estimate of the Shapley value for the feature and instance of interest.
    
    See Strumbelji and Kononenko (2014). Explanining prediction models and individual predictions with feature contributions.
        knowledge and information systems, 41(3):647-665.
    """
    
    # define the instance. shape (ncol,)
    x = X.iloc[instance_index].values 
    
    n_features = X.shape[-1]
    
    feature_indicies = [i for i in range(n_features)]

    # Generates n_iters MC estimates of the marginal contribution of a feature. 

    def create_coalitions():
         
        # select at random a permutation of indicies. shape (ncols,)
        o = np.random.permutation(feature_indicies) 
        
        # Select at random another instance. shape (ncols,)
        w = X.sample().values.flatten()
        
        x_j = x[feature_index] 
        w_j = w[feature_index]
        
        # construct two new instances
        
        b1 = np.concatenate(
            [
                np.array(x[o[:feature_index]]), # preceeding j in o from x
                np.array([x_j]), # j from x
                np.array(w[o[feature_index+1:]]) # suceeding j in o from w
            ]
        )
        
        b2 = np.concatenate(
            [
                np.array(x[o[:feature_index]]), # preceeding j in o from x
                np.array([w_j]), # j from w
                np.array(w[o[feature_index+1:]]) # suceeding j in o from w
            ]
        )
        
        
        return b1, b2
    
    # Returns a list of tuples (b1, b2)
    results = Parallel(n_jobs=n_jobs)(delayed(create_coalitions)() for i in range(n_iters))
    
    # Extracts all b1 vectors into a tuple and b2 vectors into a tuple
    b1s, b2s = zip(*results)
    
    # Converts tuples of vectors into arrays of shape (n_iters, n_features)
    b1_arr = np.stack(b1s, axis=0)
    b2_arr = np.stack(b2s, axis=0)
    
    b1_preds = np.array(estimator.predict( b1_arr ))
    b2_preds = np.array(estimator.predict( b2_arr ))
    
    # The axis across which marginal contributions should be averaged.
    # 0 for single-task models, 1 for multi-task models.
    ax = 0 
    
    if b1_preds.ndim == 3:
        # Make predictions and reshape to (targets, iterations) from (targets, iterations, 1)
        b1_preds = np.array(estimator.predict( b1_arr ))[:,:,0]
        b2_preds = np.array(estimator.predict( b2_arr ))[:,:,0]
        ax = 1
    
    marginal_contributions = b1_preds - b2_preds
    
    print(marginal_contributions.shape)
    
    # take the mean across iterations
    shapley_value = np.mean(marginal_contributions, axis=ax)
    
    return shapley_value