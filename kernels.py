import numpy as np
from collections import defaultdict
import scipy.sparse as sp

# At first, I defined a class SpectrumKernel, but useless since we can use MismatchKernel with m=0
    
class MismatchKernel:
    def __init__(self, k, m, neighbors, kmer_dict, normalize=False):
        self.k = k # length of the kmer
        self.m = m # number of maximum mismatch
        self.neighbors = neighbors # dictionnary mapping a kmer to its neighbors (explaination in vocab file)
        self.kmer_dict = kmer_dict # dictionnary mapping a kmer to its unique index
        self.normalize = normalize # if True: L2 normalization

    # Compute an embedding dictionnary for each sequance: the feature of each kmer is the count of the neighbors
    def embedding_sequence(self, sequence):
        """
        Embed a sequence as a sparse dictionary mapping feature indices to counts.
        For each k-mer in seq, add counts for all its neighbors.
        """
        embedding = defaultdict(float)
        for i in range(len(sequence) - self.k + 1): # look for each kmer in the sequence
            kmer = sequence[i:i+self.k]
            if kmer in self.kmer_dict:  # for the training, all kmer are in the dictionary, but not necessary when testing on new data
                for neighbor in self.neighbors[kmer]:
                    index_neighbor = self.kmer_dict[neighbor]
                    embedding[index_neighbor] += 1 # increment the count for this feature neighbor index

        if self.normalize: # L2 norm on the embedding
            norm_embedding = np.sqrt(sum(val**2 for val in embedding.values()))
            if norm_embedding > 0:
                for element in embedding:
                    embedding[element] = embedding[element] / norm_embedding
        return embedding

    # call the embedding_sequence function for each sequence of the dataset
    def embedding_data(self, dataset):
        embeddings = [self.embedding_sequence(sequence) for sequence in dataset]
        return embeddings

    # /!\ Need sparse matrix for the computation of SVM !! I use scipy.sparse (sp)
    def to_sparse_matrix(self, embeddings, vocab_size):
        data = []
        rows = []
        colomuns = []
        for i, embedding in enumerate(embeddings):
            for index, feature in embedding.items():
                rows.append(i)
                colomuns.append(index)
                data.append(feature)
        sparse_embeddings = sp.coo_matrix((data, (rows, colomuns)), shape=(len(embeddings), vocab_size))
        return sparse_embeddings
    

# Compute a weighted kernel of mismatch and spectrum kernels (k, m and weights are given in kernel_params)
"""
def weighted_kernel(sequences_train, sequences_test, dataset_index, kernel_params):    
    K_train_sum, K_test_sum = None, None
    for (k, m, weight) in kernel_params:
        K_train, K_test = compute_kernel_gram_for_param(sequences_train, sequences_test, k, m, dataset_index)
        if K_train_sum is None:
            K_train_sum = weight * K_train
            K_test_sum = weight * K_test
        else:
            K_train_sum += weight * K_train
            K_test_sum += weight * K_test
    return K_train_sum, K_test_sum
"""