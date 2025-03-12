import numpy as np
import pandas as pd
from kernels import MismatchKernel
from vocab import create_kmer_dict, compute_neighbors_for_dict
from svm import kernel_svm, predict_svm

# Process of each dataset separately
# Compute the predicted labels for the test set using a mismatch kernel (k,m) and a SVM
def process_dataset(dataset_index, k, m, C=1.0):
    """
    Parameters:
        dataset_index: index 0,1,2
        k: kmer length
        m: number of allowed mismatches
        C: SVM regularization parameter
    Returns:
        y_pred: predicted labels for the test set (0 or 1)
    """

    # Read the dataset of given index
    df_train = pd.read_csv(f"Kernel/data/Xtr{dataset_index}.csv", header=0, index_col=0)
    df_test = pd.read_csv(f"Kernel/data/Xte{dataset_index}.csv", header=0, index_col=0)
    df_labels = pd.read_csv(f"Kernel/data/Ytr{dataset_index}.csv", header=0, index_col=0)
    train_seqs = df_train["seq"].tolist()
    test_seqs = df_test["seq"].tolist()
    y = np.where(df_labels["Bound"].values == 0, -1, 1)

    # Build vocabulary and neighbors
    all_seqs = train_seqs + test_seqs
    kmer_dict = create_kmer_dict(all_seqs, k)
    vocab_size = len(kmer_dict)
    neighbors = compute_neighbors_for_dict(kmer_dict, m)

    # Mismatch kernel
    kernel = MismatchKernel(k, m, neighbors, kmer_dict, normalize=True)

    # Add embeddings
    train_embeddings = kernel.embedding_data(train_seqs)
    test_embeddings = kernel.embedding_data(test_seqs)

    # Convert embeddings to sparse matrices
    embedding_train_sparse = kernel.to_sparse_matrix(train_embeddings, vocab_size)
    embedding_test_sparse = kernel.to_sparse_matrix(test_embeddings, vocab_size)

    # Gram matrix for training
    K_train_sparse = embedding_train_sparse * embedding_train_sparse.T
    K_train = K_train_sparse.toarray() # 2000x2000 matrix
    
    # Train SVM on training data
    alpha, bias = kernel_svm(K_train, y, C)

    # Gram matrix for evaluation
    K_test_sparse = embedding_test_sparse * embedding_train_sparse.T
    K_test = K_test_sparse.toarray()

    # Prediction of the labels
    y_pred = predict_svm(K_test, y, alpha, bias)
    y_pred = np.where(y_pred == -1, 0, 1) # transform in binary prediction
    return y_pred