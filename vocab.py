import numpy as np
import itertools

# Build a dictionary that maps each unique substring of length k (=k-mer) found in each sequence to a unique index
def create_kmer_dict(sequences, k):
    """
    Parameters: 
        sequences : list of the DNA sequences
        k: length of the substrings to extract
    
    Returns:
        kmer_set: the dictionnary that maps k-substrings to indices
    """
    kmer_dict = {}
    index = 0
    for seq in sequences:
        for i in range(len(seq) - k + 1): # stop when there is less than k characters left
            kmer = seq[i:i+k] # the substring is the sequence between i and i+k
            if kmer not in kmer_dict:
                kmer_dict[kmer] = index # whenever we encounter a new k-substring, we add it to the dictionnary
                index += 1
    return kmer_dict

# For a given k-substring kmer, we consider that all k-substring with less than m mismatches are its neighbors
# compute_neighbors returns the list of all neighbors of a k-substring 
# /!\ this function is adapted to m=0,1,2 only 
def compute_neighbors(kmer, m):
    """
    Parameters:
        kmer : a k-substring
        m : the maximal number of mismatch
    """
    neighbors = set()
    k = len(kmer)

    # If m = 0, 1 or 2, in each case we consider the k-substring itself as a neighbor (case m=0)
    neighbors.add(kmer) 

    # If m = 1 or 2, we need to find the k-substring that have one mismatch with the original k-substring (m=1)
    if m >= 1: # first we look
        for i in range(k):
            for letter in "ACGT": # we try to replace A,C,G,T at each position
                if letter != kmer[i]:
                    neighbors.add(kmer[:i] + letter + kmer[i+1:])

    # If m = 2, we need to find the k-substring that have two mismatches with the original k-substring (m=2)
    if m >= 2:
        for i, j in itertools.combinations(range(k), 2): # same as with 1 mismatch but with pairs of positions
            for letter_i in "ACGT":
                if letter_i == kmer[i]:
                    continue
                for letter_j in "ACGT":
                    if letter_j == kmer[j]:
                        continue
                    neighbor = list(kmer)
                    neighbor[i] = letter_i
                    neighbor[j] = letter_j
                    neighbors.add(''.join(neighbor))
    return neighbors

# For each k-substring in the dictionnary, we compute the list of its (<=m)-neighbors that exist in the dictionnary
def compute_neighbors_for_dict(kmer_dict, m):
    """
    Parameters:
        kmer_dict : the dictionnary of k-substrings in the datasets
        m : the maximal number of mismatch

    Returns:
        neighbors : a dictionnary that map a k-substring to its neighbors that exist in the dictionnary
    """
    neighbors = {}
    for kmer in kmer_dict:
        all_possible_neighbors = compute_neighbors(kmer, m) # all the possible sequences that can be a neighbor

        existing_neighbors = [nbr for nbr in all_possible_neighbors if nbr in kmer_dict] # keep only those that are in the dictionnary
        neighbors[kmer] = existing_neighbors

    return neighbors