import numpy as np
import pandas as pd
from vocab import create_kmer_dict, compute_neighbors_for_dict
from kernels import MismatchKernel
from svm import kernel_svm, predict_svm

k_cv=11
m_cv=2
C_cv=1.0

def kfold_indices(n_samples, kfold):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = (n_samples // kfold) * np.ones(kfold, dtype=int)
    fold_sizes[:n_samples % kfold] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        start, stop = current, current + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx, val_idx))
        current = stop
    return folds

def cross_validate(dataset_index, C, k=k_cv, m=m_cv, kfold=5):
    df_train = pd.read_csv(f"Xtr{dataset_index}.csv", header=0, index_col=0)
    df_labels = pd.read_csv(f"Ytr{dataset_index}.csv", header=0, index_col=0)
    sequences = df_train["seq"].tolist()
    y = np.where(df_labels["Bound"].values == 0, -1, 1)
    n_samples = len(sequences)

    folds = kfold_indices(n_samples, kfold)
    train_accs = []
    eval_accs = []
    fold_nb = 1

    for train_idx, eval_idx in folds:
        seq_train = [sequences[i] for i in train_idx]
        seq_eval = [sequences[i] for i in eval_idx]
        y_train = y[train_idx]
        y_eval = y[eval_idx]

        vocab = create_kmer_dict(seq_train, k)
        vocab_size = len(vocab)

        neighbors = compute_neighbors_for_dict(vocab, m)

        kernel = MismatchKernel(k, m, neighbors, vocab, normalize=True)

        train_embeddings = kernel.embedding_data(seq_train)
        eval_embeddings = kernel.embedding_data(seq_eval)

        X_train_sparse = kernel.to_sparse_matrix(train_embeddings, vocab_size)
        X_eval_sparse = kernel.to_sparse_matrix(eval_embeddings, vocab_size)

        K_train = (X_train_sparse * X_train_sparse.T).toarray()
        K_eval = (X_eval_sparse * X_train_sparse.T).toarray()


        alpha, bias = kernel_svm(K_train, y_train, C)

        y_train_pred = predict_svm(K_train, y_train, alpha, bias)
        y_eval_pred = predict_svm(K_eval, y_train, alpha, bias)

        train_acc = np.mean(y_train_pred == y_train)
        eval_acc = np.mean(y_eval_pred == y_eval)

        print(f"Fold {fold_nb} (C={C}): training accuracy = {train_acc:.4f}, evaluation accuracy = {eval_acc:.4f}")
        train_accs.append(train_acc)
        eval_accs.append(eval_acc)
        fold_nb += 1
    avg_train = np.mean(train_accs)
    avg_eval = np.mean(eval_accs)
    print(f"Dataset {dataset_index} (C={C}, k={k}, m={m}): average training accuracy = {avg_train:.4f}, average evaluation accuracy = {avg_eval:.4f}")
    return train_accs, eval_accs


def main():
    C_cv = 1.0  
    results = {}
    for dataset_index in [0, 1, 2]:
        print(f"Dataset {dataset_index}")
        train_accs, eval_accs = cross_validate(dataset_index, C_cv, k=k_cv, m=m_cv, kfold=5)
        results[dataset_index] = (train_accs, eval_accs)

if __name__ == '__main__':
    main()