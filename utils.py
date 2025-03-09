from typing import Tuple
import numpy as np
import pandas as pd
import os
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import torch
import einops

def compute_variance_based_projection_matrix(X, y, top_k_singular=10, bias_indicator=None):
    bias_positions = np.where(y == 1)[0]
    unbias_positions = np.where(y == 0)[0]
    X_bias = X[bias_positions]
    X_unbias = X[unbias_positions]
    
    # use the smaller number of samples based on the number of samples in bias and unbias x
    min_samples = min(X_bias.shape[0], X_unbias.shape[0])
    X_bias = X_bias[:min_samples]
    X_unbias = X_unbias[:min_samples]

    # calculate the difference between bias and unbias x
    mean_bias = np.mean(X_bias, axis=0)
    mean_unbias = np.mean(X_unbias, axis=0)
    T = X_bias - X_unbias

    # centering: project out the component in the direction of unbias mean
    P = np.outer(mean_unbias, mean_unbias) / np.dot(mean_unbias, mean_unbias)
    T_centered = T @ (np.eye(T.shape[1]) - P)
    
    U, Sigma, Vt = np.linalg.svd(T_centered, full_matrices=False)
    
    bias_probes = Vt[:top_k_singular, :].T
    bias_operator = sum(np.outer(v, v) for v in bias_probes.T)  # Form projector as sum of outer products.
    
    return bias_probes, bias_operator


def load_stereoset(root_data_dir="data", type="dev"):
    data_path = os.path.join(root_data_dir, f"{type}.json")
    dataset = pd.read_json(data_path)
    dataset = dataset.iloc[0]['data']

    transformed_data = []
    for item in dataset:
        context = item['context']
        for sentence_info in item['sentences']:
            sentence = sentence_info['sentence']
            gold_label = sentence_info['gold_label']
            input_text = f"{context} {sentence}"
            label = gold_label
            target = 0 if gold_label in ['unrelated', 'anti-stereotype'] else 1
            transformed_data.append({'input': input_text, 'label': label, 'target': target, 'bias_type': item['bias_type']})
    return transformed_data


def construct_y(dataset, get_unrelated_for_detector=False):
    unrelated_indices = [i for i, d in enumerate(dataset) if d['label'] == 'unrelated']
    y = np.zeros(len(dataset))
    for i, d in enumerate(dataset):
        if get_unrelated_for_detector:
            y[i] = 0 if d['label'] == 'unrelated' else 1
        else:
            if d['label'] != 'unrelated':
                y[i] = 1 if d['label'] == 'stereotype' else 0
            else:
                y[i] = -1
    return y


def dynamic_crosscoder_hook(
    resid: torch.Tensor,  # shape [batch, seq_len, d_model]
    hook: HookPoint,
    crosscoder,
    top_indices,
    layer_num: int,
):
    B, T, d_model = resid.shape
    device = resid.device
    dtype = resid.dtype
    resid_flat = resid.reshape(B * T, d_model)
    weight_enc = crosscoder.W_enc[layer_num].to(resid.dtype)
    bias_enc = crosscoder.b_enc.to(resid.dtype)
    latent = F.relu(resid_flat @ weight_enc + bias_enc)  # shape [B*T, d_hidden]
    latent_feature = latent[:, list(top_indices)].mean(dim=-1)
    latent_feature_avg = latent_feature.reshape(B, T)  # shape [B, T]

    # construct token wise mask
    mask = torch.zeros_like(latent_feature_avg, device=device, dtype=dtype)
    mask[latent_feature_avg > 0.2] = 1
    mask[:, 0] = 0  # always set bos unactivated
    mask = mask.unsqueeze(-1).to(resid.dtype)

    reconstructed_output = einops.einsum(
        latent.reshape(B, T, -1) * 0.5,
        crosscoder.W_dec[:, layer_num],
        "batch seq_len d_hidden, d_hidden d_model -> batch seq_len d_model",
    ) + crosscoder.b_dec[layer_num]
    final_out = mask * reconstructed_output + (1 - mask) * resid

    return final_out