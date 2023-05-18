def split_sequences(sequence, n_timesteps):
    
    X, y = [], []
    seq_length = sequence.shape[0]
    for i in range(seq_length):
        
        end_ind = i + n_timesteps
        
        if end_ind > seq_length:
            break
        
        X_seq = sequence[i:end_ind, :-1]
        y_seq = sequence[end_ind-1, -1]
        
        X.append(X_seq)
        y.append(y_seq)
    
    return np.array(X),np.array(y)
