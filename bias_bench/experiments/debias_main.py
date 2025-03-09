import argparse
import json
import sys
import os
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss

import transformers
thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(thisdir, '..'))
sys.path.append(os.path.join(thisdir, '../..'))
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.benchmark.seat import SEATRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

from bias_bench.debias.self_debias.self_debiasing import (
    DEBIASING_PREFIXES,
    DEBIASING_KEYWORDS,
)
DEBIASING_PREFIXES = {
    "race-color": "The following text discriminates against people because of their race/color: ",
    "gender": "The following text discriminates against people because of their gender: ",
    "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
    "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
    "religion": "The following text discriminates against people because of their religion: ",
    "age": "The following text discriminates against people because of their age: ",
    "nationality": "The following text discriminates against people because of their nationality: ",
    "disability": "The following text discriminates against people because of their disability: ",
    "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs Debias Main benchmark.")
######################
# General Arguments
######################
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="GeneralTLModel",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt2-xl",
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--is_tl_model",
    type=str2bool,
    nargs='?',
    const=True,
    default=True,
)
parser.add_argument(
    "--method",
    action="store",
    type=str,
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Device to use (e.g., cuda, cpu).",
)
######################
# Steroset Arguments
######################
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=64,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)
parser.add_argument(
    "--alpha",
    action="store",
    type=float,
    default=0.1,
    help="alpha for debias",
)
######################
# SEAT Arguments
######################
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=100000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
######################
# Perplexity Arguments
######################
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where results are written.",
)
parser.add_argument(
        "--decay_constant",
        type=float,
        default=50,
        help="Value for the decay constant (lambda in the paper)",
    )
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="Minimum factor by which each probability is multiplied",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=2048,
    help="The maximum input length to be processed (-1 corresponds to the model's context window)",
)
parser.add_argument(
    "--max_length_pattern",
    type=int,
    default=32,
    help="The number of tokens to reserve for the self-diagnosis patterns",
)
parser.add_argument(
    "--stride",
    type=int,
    default=-1,
    help="If set, for the first --stride tokens no loss is computed",
)
parser.add_argument(
    "--use_keywords",
    action="store_true",
    help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, additional debugging output is printed to stdout",
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument("--bias_type", action="store", type=str, default="gender")
parser.add_argument(
    "--is_self_debias",
    action="store_true",
)


if __name__ == "__main__":
    args = parser.parse_args()

    print("---- Running Debias Main exp: ----")
    print("---- General Arguments:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - is_tl_model: {args.is_tl_model}")
    print(f" - method: {args.method}")
    print(f" - device: {args.device}")
    print("---- Stereoset Arguments:")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")
    print(f" - alpha: {args.alpha} (Not used)")
    print("---- SEAT Arguments:")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print("---- Perplexity Arguments:")
    print(f" - output_dir: {args.output_dir}")
    print(f" - decay_constant: {args.decay_constant}")
    print(f" - epsilon: {args.epsilon}")
    print(f" - max_length: {args.max_length}")
    print(f" - max_length_pattern: {args.max_length_pattern}")
    print(f" - stride: {args.stride}")
    print(f" - use_keywords: {args.use_keywords}")

    model = getattr(models, args.model)(args.model_name_or_path, args.method, device=args.device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    run_perplexity = True
    run_seat = True
    run_stereoset = True

    ######################
    # Perplexity
    ######################
    if run_perplexity:
        print("Running Perplexity benchmark:")
            # Override loaded the model.
        kwargs = {}
        if args.bias_direction is not None:
            # Load the pre-computed bias direction for SentenceDebias.
            bias_direction = torch.load(args.bias_direction)
            kwargs["bias_direction"] = bias_direction

        if args.projection_matrix is not None:
            # Load the pre-computed projection matrix for INLP.
            projection_matrix = torch.load(args.projection_matrix)
            kwargs["projection_matrix"] = projection_matrix

        device = args.device

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        if args.is_self_debias:
            # Move the model to GPU, if available.
            model._model.to(device)

            max_length = (
                args.max_length if args.max_length > 0 else model._model.config.n_positions
            ) - args.max_length_pattern
        else:
            model.to(device)
            max_length = (
                args.max_length if args.max_length > 0 else model.config.max_position_embeddings if not args.is_tl_model else model.cfg.n_ctx
            ) - args.max_length_pattern

        if args.stride <= 0:
            args.stride = max_length

        lls = []
        ppl = None
        os.makedirs(args.output_dir, exist_ok=True)
        print_step = 0

        with open(f"{args.output_dir}/{args.method}.txt", "w", encoding="utf8") as fh:
            fh.write(f"=== RESULT [{args.model}] ===\n")

            for i in tqdm(range(0, encodings.input_ids.size(1), args.stride)):
                begin_loc = max(i + args.stride - max_length, 0)
                end_loc = min(i + args.stride, encodings.input_ids.size(1))
                trg_len = end_loc - i  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                debiasing_prefixes = [DEBIASING_PREFIXES[args.bias_type]]

                with torch.no_grad():
                    if args.is_self_debias:
                        loss = model.compute_loss_self_debiasing(
                            input_ids=input_ids,
                            trg_len=trg_len,
                            debiasing_prefixes=debiasing_prefixes,
                            decay_constant=args.decay_constant,
                            epsilon=args.epsilon,
                            debug=args.debug,
                        )

                    else:
                        # lm_logits = outputs[1]
                        lm_logits = model(input_ids, labels=target_ids)[1] if not args.is_tl_model else model(input_ids, tokens=target_ids)

                        # Shift so that tokens < n predict n
                        shift_logits = lm_logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()
                        # Flatten the tokens
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                        )

                    log_likelihood = loss * trg_len

                lls.append(log_likelihood)

                ppl = torch.exp(torch.stack(lls).sum() / end_loc)

                if print_step % 100 == 0:
                    print(f"token {i}: {ppl}")
                print_step += 1

                fh.write(f"token {i}: {ppl}\n")

            print(f"Final perplexity: {ppl}")
            fh.write(f"=== FINAL RESULT [{args.method}] ===\n")
            fh.write(f"Perplexity: {ppl}\n")

    ######################
    # SEAT
    ######################
    if run_seat:
        experiment_id = generate_experiment_id(
            name="seat", model=args.model, model_name_or_path=args.model_name_or_path
        )
        print("Running SEAT benchmark:")
        runner = SEATRunner(
            experiment_id=experiment_id,
            tests=args.tests,
            data_dir=f"{args.persistent_dir}/data/seat",
            n_samples=args.n_samples,
            parametric=args.parametric,
            model=model if args.is_tl_model else model.transformer if 'gpt' in args.model_name_or_path else model.model,
            tokenizer=tokenizer,
            is_tl_model=args.is_tl_model,
        )
        results = runner()
        print(results)
        os.makedirs(f"{args.persistent_dir}/results/seat", exist_ok=True)
        with open(f"{args.persistent_dir}/results/seat/{args.model}_{args.method}.json", "w") as f:
            json.dump(results, f)

    ######################
    # Steroset
    ######################
    if run_stereoset:
        experiment_id = generate_experiment_id(
            name="stereoset", model=args.model, model_name_or_path=args.model_name_or_path
        )
        print("Running StereoSet:")
        runner = StereoSetRunner(
            intrasentence_model=model,
            tokenizer=tokenizer,
            input_file=f"{args.persistent_dir}/data/stereoset/test.json",
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            is_generative=_is_generative(args.model),
            debias=True,
            alpha=args.alpha,
            is_tl_model=args.is_tl_model,
            device=args.device,
        )
        results = runner()
        # print(results)
        os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
        with open(
            f"{args.persistent_dir}/results/stereoset/{args.model}_{args.method}.json", "w"
        ) as f:
            json.dump(results, f, indent=2)


    ######################
    # 
    ######################