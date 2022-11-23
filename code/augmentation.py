import torch
import numpy as np


def build_mask(gene_num, masked_percentage, device):
    mask = torch.cat([torch.ones(int(gene_num * masked_percentage), dtype=bool),
                      torch.zeros(gene_num - int(gene_num * masked_percentage), dtype=bool)])
    shuffle_index = torch.randperm(gene_num)
    return mask[shuffle_index].to(device)


def random_mask(data, mask_percentage, apply_mask_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_mask_prob:
        mask = build_mask(data.shape[1], mask_percentage, device)
        data[:, mask] = 0
    return data


def random_gaussian_noise(data, noise_percentage, sigma, apply_noise_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_noise_prob:
        mask = build_mask(data.shape[1], noise_percentage, device)
        noise = torch.randn(int(data.shape[1] * noise_percentage)) * sigma
        data[:, mask] += noise.to(device)
    return data


def random_swap(data, swap_percentage, apply_swap_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_swap_prob:
        swap_instances = int(data.shape[1] * swap_percentage / 2)
        swap_pair = torch.randint(data.shape[1], size=(swap_instances, 2)).to(device)
        data[:, swap_pair[:, 0]], data[:, swap_pair[:, 1]] = data[:, swap_pair[:, 1]], data[:, swap_pair[:, 0]]
    return data


def instance_crossover(data, cross_percentage, apply_cross_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_cross_prob:
        cross_idx = torch.randint(data.shape[0], size=(1, )).to(device)
        cross_instance = data[cross_idx]
        mask = build_mask(data.shape[1], cross_percentage, device)
        data[:, mask] = cross_instance[:, mask]
    return data


def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


def pretext_generator(m, x):
    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def transformation(data, mask_percentage=0.1, apply_mask_prob=0.5, noise_percentage=0.1, sigma=0.5, apply_noise_prob=0.5,
                   swap_percentage=0.1, apply_swap_prob=0.5, cross_percentage=0.1, apply_cross_prob=0.5, device=torch.device("cuda")):
    data = random_mask(data, mask_percentage, apply_mask_prob, device)
    data = random_gaussian_noise(data, noise_percentage, sigma, apply_noise_prob, device)
    # data = random_swap(data, swap_percentage, apply_swap_prob, device)
    # data = instance_crossover(data, cross_percentage, apply_cross_prob, device)
    return data



