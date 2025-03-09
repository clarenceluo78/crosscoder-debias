from utils import *
from transformer_lens import ActivationCache
import tqdm
from warnings import warn

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder.

    I swap out the buffer completely when it runs out instead of swapping out some subset
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        if model_B is not None:
            assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        tot_tokens_initial_estimate = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = tot_tokens_initial_estimate // ((cfg["seq_len"] - 1) * cfg["model_batch_size"])
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        print(f"We will have {self.buffer_size} tokens in the buffer")
        
        self.layer_start, self.layer_end = cfg["layer_length"]
        self.layer_len = self.layer_end - self.layer_start
        if self.layer_len == 0:
            self.layer_len = 2

        self.buffer = torch.zeros(
            (self.buffer_size, self.layer_len, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding layer_len for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens

        if self.layer_start == self.layer_end and self.model_B is not None:
            estimated_norm_scaling_list = []
            for model in [model_A, model_B]:
                estimated_norm_scaling_factor = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model, layer_start=self.layer_start)
                estimated_norm_scaling_list.append(estimated_norm_scaling_factor)
        else:
            estimated_norm_scaling_list = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A, layer_start=self.layer_start, layer_end=self.layer_end, is_same_model=True)

        print(f"Estimated norm scaling for each layer: {estimated_norm_scaling_list}")
        self.normalisation_factor = torch.tensor(
            estimated_norm_scaling_list,
            device=self.cfg["device"],
            dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(
        self, 
        batch_size, 
        model, 
        layer_start,
        layer_end,
        n_batches_for_norm_estimate: int = 50,
        is_same_model: bool = False,
    ):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            if is_same_model:
                hook_points = [f"blocks.{layer}.{self.cfg['hook_point']}" for layer in range(layer_start, layer_end)]
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=hook_points,
                    return_type=None,
                )
                norms_per_batch_per_layer = []
                for hook_point in hook_points:
                    acts = cache[hook_point]
                    acts = acts[:, 1:, :]
                    norms_per_batch_per_layer.append(acts.norm(dim=-1).mean().item())
                norms_per_batch.append(norms_per_batch_per_layer)
            else:
                hook_point = f"blocks.{layer_start}.{self.cfg['hook_point']}"
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=hook_point,
                    return_type=None,
                )
                acts = cache[self.cfg["hook_point"]]
                acts = acts[:, 1:, :]
                norms_per_batch.append(acts.norm(dim=-1).mean().item())
        if is_same_model:
            mean_norms = np.mean(norms_per_batch, axis=0)
            scaling_factors = np.sqrt(model.cfg.d_model) / mean_norms
            return scaling_factors
        else:
            mean_norm = np.mean(norms_per_batch)
            scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm
            return scaling_factor

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer")
        self.pointer = 0
        self.buffer = torch.zeros(
            (self.buffer_size, self.layer_len, self.model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(self.cfg["device"])
        torch.cuda.empty_cache()
        with torch.autocast("cuda", torch.bfloat16):
            self.first = False
            for i in range(0, self.buffer_batches):
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + self.cfg["model_batch_size"]
                ]
                assert tokens.shape == (
                    self.cfg["model_batch_size"],
                    self.cfg["seq_len"],
                )
                if self.layer_start == self.layer_end and self.model_B is not None:
                    # Use activations from both models, as done when estimating norm scaling.
                    hook_point = self.cfg["hook_point"]
                    _, cache_A = self.model_A.run_with_cache(tokens, names_filter=hook_point)
                    _, cache_B = self.model_B.run_with_cache(tokens, names_filter=hook_point)
                    acts_A = cache_A[hook_point][:, 1:, :]  # Drop BOS
                    acts_B = cache_B[hook_point][:, 1:, :]
                    # Stack across a new dimension to mimic two layers.
                    acts = torch.stack([acts_A, acts_B], dim=0)  # shape: (2, batch, seq_len-1, d_model)
                    acts = einops.rearrange(
                        acts, "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model"
                    )
                else:
                    # Use multiple hook points (from layer_start to layer_end) from model_A only,
                    # following similar logic as in estimate_norm_scaling_factor.
                    hook_points = [
                        f"blocks.{layer}.{self.cfg['hook_point']}"
                        for layer in range(self.layer_start, self.layer_end)
                    ]
                    _, cache_A = self.model_A.run_with_cache(tokens, names_filter=hook_points)
                    acts_list = []
                    for hp in hook_points:
                        acts_single = cache_A[hp][:, 1:, :]  # Drop BOS token
                        acts_list.append(acts_single)
                    # Stack along a new dimension to correspond to each layer.
                    acts = torch.stack(acts_list, dim=1)  # shape: (batch, n_layers, seq_len-1, d_model)
                    acts = einops.rearrange(
                        acts, "batch n_layers seq_len d_model -> (batch seq_len) n_layers d_model"
                    )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        # Check that the buffer is filled
        assert self.buffer.shape[0] == self.pointer, (
            f"Buffer size {self.buffer.shape} does not match pointer {self.pointer}"
        )
        assert (
            torch.sum(self.buffer[-1].abs()) > 1
        ), f"Last batch in buffer is not refreshed \n\n {self.buffer[-1]}"

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer + self.cfg["batch_size"] > self.buffer.shape[0]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out