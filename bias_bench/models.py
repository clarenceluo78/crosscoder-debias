from functools import partial
import joblib
import torch
import sys
sys.append('..')
from utils import compute_variance_based_projection_matrix, load_stereoset, dynamic_crosscoder_hook, construct_y
from crosscoder import CrossCoder

import os

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import transformers
from transformer_lens import HookedTransformer
import pickle
import numpy as np
from functools import partial
import transformer_lens.utils as tl_utils
from functools import partial
from pathlib import Path


# we can return a hooked model for downstream tasks with crosscoder loaded (hooked)

class GeneralTLModel:
    def __new__(self, model_name_or_path, method="crosscoder", device="cuda"):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        tl_model = HookedTransformer.from_pretrained(
            model_name_or_path, hf_model=hf_model,
            torch_dtype=torch.float32 if "gpt" in model_name_or_path else torch.bfloat16,
            fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False,
        )
        tl_model = tl_model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        generate_samples = False

        dataset = load_steroset(type='dev')

        unrelated_indices = [i for i, d in enumerate(dataset) if d['label'] == 'unrelated']
        gender_bias_indices = [i for i, d in enumerate(dataset) if d['label'] != 'unrelated' and d['bias_type'] == 'gender'] 
        profession_bias_indices = [i for i, d in enumerate(dataset) if d['label'] != 'unrelated' and d['bias_type'] == 'profession'] 
        race_bias_indices = [i for i, d in enumerate(dataset) if d['label'] != 'unrelated' and d['bias_type'] == 'race'] 
        religion_bias_indices = [i for i, d in enumerate(dataset) if d['label'] != 'unrelated' and d['bias_type'] == 'religion'] 

        model_name = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path
        output_file_name = model_name + "_" + method

        if generate_samples:
            usual_prompt = "The capital name of France is"
            bias_prompt = "Asian students are expected to be good at"
            output_dir = "/homes/hl5723/Workspace/evaluation/bias-bench/results/generation_samples"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{output_file_name}.txt")
            if os.path.exists(output_file):
                open(output_file, "w").close()
            with open(output_file, "a") as f:
                f.write(f"Original Model\n")
                
                f.write(f"Usual Prompt: {usual_prompt}\n")
                f.write(f"Generated Text: {tl_model.generate(usual_prompt, max_new_tokens=30, top_k=10, top_p=0.95, temperature=0)}\n\n")
                
                f.write(f"Bias Prompt: {bias_prompt}\n")
                f.write(f"Generated Text: {tl_model.generate(bias_prompt, max_new_tokens=30, top_k=10, top_p=0.95, temperature=0)}\n\n")
        
    
        X_for_variance = np.load(f"/data2/hluo/dataset/stereoset/{model_name}/X_25_resid_pre_latercontext.npy")
        X_for_variance = np.delete(X_for_variance, unrelated_indices, axis=0)
        y = construct_y(dataset, get_unrelated_for_detector=False)
        bias_probes, bias_operator = compute_variance_based_projection_matrix(X_for_variance, y, top_k_singular=10)
        
        crosscoder_layers = CrossCoder.load(
            "version_0",
            checkpoint_version=0, 
            save_dir="/data2/hluo/checkpoints/crosscoders/qwen_layer_14_24"
        )
        crosscoder_layers.to(device)
        crosscoder_layers.eval()
        crosscoder_layers = crosscoder_layers.to(tl_model.cfg.dtype)

        latent_vectors = crosscoder_layers.W_dec.mean(dim=1)
        latent_vectors = latent_vectors / torch.norm(latent_vectors, dim=1, keepdim=True)

        num_top_latents = 20
        bias_probes_matrix = torch.tensor(bias_probes, dtype=tl_model.cfg.dtype).to(device)
        bias_probes_matrix = bias_probes_matrix / bias_probes_matrix.norm(dim=0, keepdim=True)
        latent_vectors = crosscoder_layers.W_dec.mean(dim=1)  # shape: [num_latents, d_in]
        latent_vectors_norm = latent_vectors / latent_vectors.norm(dim=-1, keepdim=True)
        proj_coeffs = latent_vectors_norm @ bias_probes_matrix  # shape: [num_latents, num_probes]
        alignment_scores = torch.linalg.norm(proj_coeffs, dim=1)  # shape: [num_latents]
        alignment_scores_np = alignment_scores.detach().cpu().float().numpy()
        top_indices = np.argsort(alignment_scores_np)[-num_top_latents:][::-1]
        top_info = []
        for idx in top_indices:
            latent_vector = latent_vectors[idx].detach().float().cpu().numpy()
            alignment_score = alignment_scores_np[idx]
            top_info.append((idx, alignment_score))
        print("Top 20 latent indices by alignment score:", top_info)
        
        layer_start = 14
        layer_end = 24
        hook_point_name = "hook_resid_pre" 
        for i, layer_num in enumerate(range(layer_start, layer_end)):

            tl_model.add_hook(
                tl_utils.get_act_name('resid_pre', layer_num),
                partial(
                    dynamic_crosscoder_hook,
                    crosscoder=crosscoder_layers,
                    top_indices=top_indices,
                    layer_num=i,
                ),
                is_permanent=True,
                prepend=True,
            )

        if generate_samples:
            # 3h. Log sample generations.
            with open(output_file, "a") as f:
                f.write(f"\n--- Layer {layer_num} Debias + Patching ---\n")
                f.write(f"Usual Prompt: {usual_prompt}\n")
                f.write(f"Generated: {tl_model.generate(usual_prompt, max_new_tokens=30, top_k=10, top_p=0.95, temperature=0)}\n\n")
                f.write(f"Bias Prompt: {bias_prompt}\n")
                f.write(f"Generated: {tl_model.generate(bias_prompt, max_new_tokens=30, top_k=10, top_p=0.95, temperature=0)}\n\n")

        return tl_model


class GeneralHFModel:
    def __new__(self, model_name_or_path, method=None, device="cuda"):
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32 if "gpt" in model_name_or_path else torch.bfloat16).to(device) if "gpt" in model_name_or_path \
              else AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32 if "gpt" in model_name_or_path else torch.bfloat16).to(device)