# # Train correctness probe

# ## Requierments:
# 1. prompts the model to answer questions from `dataset`
# 2. record question's last token actiavtion from mid layers (50%)
# 3. Label every question as correct/ incorrect depending whether the model got it right or not
#     1.  Uses LLM as a judge
# 4. partition the activations to correct and incorrect sets
# 5. Calculates the direction from the correct group to the incorrect group
#     1. Translate vectors to have as origin the mean of the two group's centriods, call it $o$
#     2. Calcluate the mean direction from $o$ to correct, $\mu_{\text{correct}}$
#     3. Calcluate the mean direction from $o$ to incorrect, $\mu_{\text{incorrect}}$
#     4. return $0.5 \cdot (\mu_{\text{correct}} - \mu_{\text{incorrect}})$
# 6. Construct scorer class with
#     1. The direction from incorret to corret $\mu$
#     2. The new origin vector $o$
#     3. A score method that computes for hidden state $h$, score(h) $= \frac{1}{\lVert \mu \rVert}\mu^T \cdot \left(h - o \right)$

# ## Setup


# Run before imports so that HF_HOME is set
import os
import gc
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import openai
import re
from adais.adaptive.probe import CorrectnessScorer
from adais.datasets import gsm8k
from adais.datasets import math_dataset
from adais.datasets import mmlu_pro
from adais.datasets import bbh


print(f"CUDA available: \t{torch.cuda.is_available()}")
print(f"Torch version: \t\t{torch.__version__}")
if torch.cuda.is_available():
    print(f"Number of CUDA devices\t {torch.cuda.device_count()}")
    print(f"CUDA device:\t\t {torch.cuda.get_device_name(torch.cuda.current_device())}")


@dataclass
class config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # "google/gemma-2-2b-it"  hf model
    judge_model_name: str = "gpt-4o-mini"  # model used for judging answers
    base_save_dir: str = "/home/yandex/APDL2425a/group_12/gorodissky/AdaIS/output"  # base directory to save outputs
    dataset_name: str = "MMLU"  # options are "MMLU", "GSM8K", "BBH", "MATH"
    dataset_size: int = 16
    seed: int = 1337
    batch_size: int = 2


# ## Load model and dataset
def load_model_and_dataset():
    global cfg
    # Load dataset
    print(f"Loading dataset: {cfg.dataset_name}")
    match cfg.dataset_name:
        case "MMLU":
            ds = mmlu_pro.get_dataset()
        case "GSM8K":
            ds = gsm8k.get_dataset()
        case "BBH":
            ds = bbh.get_dataset()
        case "MATH":
            ds = math_dataset.get_dataset()
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")
    size = min(cfg.dataset_size, len(ds.data))
    ds.data = ds.data.sample(size, random_state=cfg.seed, ignore_index=True)
    ds.data["question"] = ds.data["question"].apply(ds.format_question)
    print(f"Dataset loaded: {len(ds.data)} examples")

    # Load model and tokenizer
    print(f"\nLoading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype="auto",
        device_map="auto",
    )
    # Set pad token if not set
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on device: {model.device} dtype: {model.dtype}")
    print(f"Model has {model.config.num_hidden_layers} layers")
    return model, tokenizer, ds


# ## Set up activation capture for mid layers (50%) and neighbors
def cleanup_hooks(model):
    num_layers = model.config.num_hidden_layers
    for layer_indx in range(num_layers):
        model.model.layers[layer_indx]._forward_hooks.clear()


def register_activation_hooks(
    model,
    activations_storage_minus1,
    activations_storage_mid,
    activations_storage_plus1,
):
    # First cleanup
    cleanup_hooks(model)
    mid_layer_idx = model.config.num_hidden_layers // 2
    layer_indices = [mid_layer_idx - 1, mid_layer_idx, mid_layer_idx + 1]
    print(f"Capturing activations from layers {layer_indices} (around 50% depth)")

    def activation_hook_minus1(module, input, output):
        """Hook to capture the last token's hidden state from layer mid-1"""
        if isinstance(output, tuple):
            output = output[0]  # if output is a tuple, first element is hidden states
        last_token_activation = output[:, -1, :].detach().cpu()
        activations_storage_minus1.append(last_token_activation)

    def activation_hook_mid(module, input, output):
        """Hook to capture the last token's hidden state from middle layer"""
        if isinstance(output, tuple):
            output = output[0]  # if output is a tuple, first element is hidden states
        last_token_activation = output[:, -1, :].detach().cpu()
        activations_storage_mid.append(last_token_activation)

    def activation_hook_plus1(module, input, output):
        """Hook to capture the last token's hidden state from layer mid+1"""
        if isinstance(output, tuple):
            output = output[0]  # if output is a tuple, first element is hidden states
        last_token_activation = output[:, -1, :].detach().cpu()
        activations_storage_plus1.append(last_token_activation)

    # Register hooks on the three layers
    hook_handle_minus1 = model.model.layers[mid_layer_idx - 1].register_forward_hook(
        activation_hook_minus1
    )
    hook_handle_mid = model.model.layers[mid_layer_idx].register_forward_hook(
        activation_hook_mid
    )
    hook_handle_plus1 = model.model.layers[mid_layer_idx + 1].register_forward_hook(
        activation_hook_plus1
    )

    return np.array(layer_indices)


# ## Generate responses and collect activations
def generate_and_collect_activations(
    model,
    tokenizer,
    ds,
    activations_storage_minus1,
    activations_storage_mid,
    activations_storage_plus1,
):
    global cfg
    print(f"\nGenerating responses with batch_size={cfg.batch_size}...")

    all_activations = []

    df = ds.data
    df["answer"] = None
    df["raw_answer"] = None
    for i in tqdm(range(0, len(df), cfg.batch_size), desc="Batches"):
        batch = df.iloc[
            i : i + cfg.batch_size
        ]  # may be smaller than batch_size in the last batch

        # Prepare questions
        questions = batch[
            "question"
        ].tolist()  # question are already formatted as prompts

        # Format prompts
        prompts = [[{"role": "user", "content": q}] for q in questions]

        # Tokenize
        tokenizer.padding_side = "left"  # last token is the prompt's
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # # DEBUG: print proccessed prompt
        # print(f"inputs_ids shape: {inputs['input_ids'].shape}")
        # print("attention mask:\n", inputs["attention_mask"])
        # proccessed_prompts = tokenizer.batch_decode(
        #     inputs["input_ids"], skip_special_tokens=False
        # )
        # print("proccessed prompts:")
        # for pp in proccessed_prompts:
        #     print("=" * 120)
        #     print(pp)

        # Clear previous activations
        activations_storage_minus1.clear()
        activations_storage_mid.clear()
        activations_storage_plus1.clear()

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # greedy decoding matching the paper's setup
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # # DEBUG: print responses
        # print(type(outputs))
        # answers_tokens = outputs[:, inputs["input_ids"].size(1):]
        # answers = tokenizer.batch_decode(answers_tokens, skip_special_tokens=False)
        # for a in answers:
        #     print(a)
        # break

        # remove prompt
        prompt_lengths = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_lengths:]
        answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        df.iloc[i : i + cfg.batch_size, df.columns.get_loc("raw_answer")] = answers

        answers = [ds.extract_answer_func(a)[0] for a in answers]

        # Set answers in df
        answer_col_idx = df.columns.get_loc("answer")
        df.iloc[i : i + cfg.batch_size, answer_col_idx] = answers

        # Store activations from the FIRST forward pass (the prompt processing)
        # The first forward pass processes the full prompt and the hooks captures the last token activation
        # The subsequent forward passes proccess the new generated token only, each time sequence of length 1
        if (
            activations_storage_minus1
            and activations_storage_mid
            and activations_storage_plus1
        ):
            act_minus1 = activations_storage_minus1[0]
            act_mid = activations_storage_mid[0]
            act_plus1 = activations_storage_plus1[0]

            # Compute mean across the 3 layers
            mean_activation = torch.stack([act_minus1, act_mid, act_plus1], dim=0).mean(
                dim=0
            )
            all_activations.append(mean_activation)

        # Release GPU memory
        del inputs, outputs, generated_tokens
        gc.collect()
        torch.cuda.empty_cache()

    # Set is correct column
    def normalize_str(s):
        if pd.isna(s):
            return ""
        return re.sub(r"\W", "", s)

    df["is_correct"] = df.apply(
        lambda row: normalize_str(row["answer"]) == normalize_str(row["golden_label"]),
        axis=1,
    )
    # Remove hooks
    cleanup_hooks(model)

    # Concatenate all activations
    activations_tensor = torch.cat(all_activations, dim=0)
    print(f"Done generating generating answers for {len(df)} questions ")
    print(
        f"Activations shape: {activations_tensor.shape} (Mean of middle layers from prompt's last token)"
    )
    return activations_tensor


# ## LLM as a judge to label correct/incorrect


def label_with_judge(all_questions, all_ground_truths, all_model_answers):
    global cfg
    with open(
        "/home/yandex/APDL2425a/group_12/gorodissky/tokens/openai_key_personal.txt", "r"
    ) as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

    def query_gpt(prompt, model="gpt-4.1-nano"):
        client = openai.OpenAI()

        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        return text

    print("\nEvaluating answers using LLM as judge...")

    judge_prompt_template = """You are evaluating whether a model's answer is correct.

    Question: {question}
    Ground Truth: {ground_truth}
    Model Answer: {model_answer}

    Is the model's answer correct? Consider semantic equivalence, not just exact match. Think step by step.
    Respond with only "CORRECT" or "INCORRECT".

    Judgment:"""

    correctness_labels = []

    for i, (q, gt, ma) in enumerate(
        tqdm(
            zip(all_questions, all_ground_truths, all_model_answers),
            total=len(all_questions),
            desc="Judging",
        )
    ):
        judge_prompt = judge_prompt_template.format(
            question=q, ground_truth=gt, model_answer=ma
        )
        judgment = query_gpt(judge_prompt, model=cfg.judge_model_name)
        is_correct = (
            "CORRECT" in judgment.upper() and "INCORRECT" not in judgment.upper()
        )
        # # DEBUG: print prompt and judgment
        # print("\n--- Judge Prompt ---")
        # print(judge_prompt)
        # print("--- Judgment ---")
        # print(judgment)
        # print(f"Is correct: {is_correct}")
        correctness_labels.append(is_correct)

    correctness_array = np.array(correctness_labels)
    num_correct = correctness_array.sum()
    num_incorrect = len(correctness_array) - num_correct

    print(
        f"\nCorrect: {num_correct} ({100 * num_correct / len(correctness_array):.1f}%)"
    )
    print(
        f"Incorrect: {num_incorrect} ({100 * num_incorrect / len(correctness_array):.1f}%)"
    )
    return correctness_array


# ## Calculate correctness direction


def calculate_correctness_direction(
    acts: np.ndarray,
    labels: np.ndarray,
    layer_indices: np.ndarray,
) -> CorrectnessScorer:
    print("\nCalculating probe direction...")

    correct_acts = acts[labels]
    incorrect_acts = acts[~labels]

    print(f"Correct activations: {correct_acts.shape}")
    print(f"Incorrect activations: {incorrect_acts.shape}")

    # Calculate centroids
    correct_centroid = correct_acts.mean(axis=0)
    incorrect_centroid = incorrect_acts.mean(axis=0)

    # New origin is the midpoint between centroids
    new_origin = (correct_centroid + incorrect_centroid) / 2

    # Direction from incorrect to correct
    direction = correct_centroid - incorrect_centroid

    # Calculate quantiles on validation set
    scorer = CorrectnessScorer(direction, new_origin, layer_indices)

    print(f"\nDirection vector shape: {direction.shape}")
    print(f"Direction vector norm: {np.linalg.norm(direction):.4f}")

    return scorer


# ## Calculate metadata


def calculate_metadata(
    scorer,
    acts,
    labels,
    layer_indices,
):
    global cfg
    print("Generating metadata...")

    # Score all activations
    scores = scorer.score(acts)

    def get_stats(scores, labels):
        num_correct = labels.sum()
        num_incorrect = (~labels).sum()

        correct_scores = scores[labels]
        incorrect_scores = scores[~labels]

        mean_correct = correct_scores.mean()
        mean_incorrect = incorrect_scores.mean()
        std_correct = correct_scores.std()
        std_incorrect = incorrect_scores.std()

        return (
            mean_correct,
            std_correct,
            num_correct,
            mean_incorrect,
            std_incorrect,
            num_incorrect,
        )

    (
        mean_score_correct,
        std_score_correct,
        num_correct,
        mean_score_incorrect,
        std_score_incorrect,
        num_incorrect,
    ) = get_stats(scores, labels)

    # Calculate separation
    separation = np.abs(mean_score_correct - mean_score_incorrect)
    # Simple threshold classification (at midpoint)
    threshold = (mean_score_correct + mean_score_incorrect) / 2

    predicted_correct = scores > threshold
    accuracy = (predicted_correct == labels).mean()
    print(f"Accuracy: {accuracy:.2%}")

    metadata = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "dataset_size": cfg.dataset_size,
        "seed": cfg.seed,
        "layers_idx": layer_indices.tolist(),
        "num_correct": int(num_correct),
        "correct_score_mean": float(mean_score_correct),
        "correct_score_std": float(std_score_correct),
        "num_incorrect": int(num_incorrect),
        "incorrect_score_mean": float(mean_score_incorrect),
        "incorrect_score_std": float(std_score_incorrect),
        "separation": float(separation),
        "threshold (midpoint between means of train set)": float(threshold),
        "accuracy": float(accuracy),
    }
    return metadata


# ## Save
def save(scorer, metadata):
    global cfg
    save_dir = Path(cfg.base_save_dir)
    save_dir = save_dir / "probe_results" / cfg.dataset_name / cfg.model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the scorer
    version = datetime.now().strftime("%Y-%m-%d_%H:%M")
    scorer_path = save_dir / f"correctness_scorer_{version}.npz"
    scorer.save(str(scorer_path))

    # Save metadata and results

    metadata_path = save_dir / f"metadata_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f" Results saved to {save_dir}")
    print(f"  - Scorer: {scorer_path}")
    print(f"  - Metadata: {metadata_path}")


# # Run whole pipline


def run():
    global cfg
    cfg = config()
    cfg.dataset_size = int(2 ** 13)
    cfg.batch_size = 32
    for model_name in [
        # "Qwen/Qwen2.5-0.5B-Instruct",
        # "google/gemma-2-9b-it",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "mistralai/Ministral-8B-Instruct-2410",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ]:
        print(f"\n\n=== Processing model: {model_name} ===")
        cfg.model_name = model_name
        model, tokenizer, ds = load_model_and_dataset()

        # Prepare activation storage
        activations_storage_minus1: List[torch.Tensor] = []
        activations_storage_mid: List[torch.Tensor] = []
        activations_storage_plus1: List[torch.Tensor] = []

        # Register hooks
        layer_indices = register_activation_hooks(
            model,
            activations_storage_minus1,
            activations_storage_mid,
            activations_storage_plus1,
        )

        # Generate and collect activations
        acts = generate_and_collect_activations(
            model,
            tokenizer,
            ds,
            activations_storage_minus1,
            activations_storage_mid,
            activations_storage_plus1,
        )
        acts = acts.to(torch.float32).numpy()
        labels = ds.data["is_correct"].copy().to_numpy()

        # # Label with judge
        # correctness_array = label_with_judge(
        #     all_questions, all_ground_truths, all_model_answers
        # )

        # Calculate correctness direction
        scorer = calculate_correctness_direction(acts, labels, layer_indices)

        # Calculate metadata
        metadata = calculate_metadata(
            scorer,
            acts,
            labels,
            layer_indices,
        )

        # Save results
        save(scorer, metadata)


if __name__ == "__main__":
    run()
