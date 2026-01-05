
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

os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
print(f"HF_HOME set to:\t\t {os.environ['HF_HOME']}")

import torch
from typing import Tuple, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import openai
from cisc.src.diis.probe import CorrectnessScorer

print(f"CUDA available: \t{torch.cuda.is_available()}")
print(f"Torch version: \t\t{torch.__version__}")
if torch.cuda.is_available():
    print(f"Number of CUDA devices\t {torch.cuda.device_count()}")
    print(f"CUDA device:\t\t {torch.cuda.get_device_name(torch.cuda.current_device())}")

@dataclass
class config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # "google/gemma-2-2b-it"  hf model
    judge_model_name: str = "gpt-4o-mini"  # model used for judging answers
    base_save_dir: str = "/home/yandex/APDL2425a/group_12/gorodissky/google-research/cisc/output"  # base directory to save outputs
    dataset_name: tuple[str, str] = (
        "mandarjoshi/trivia_qa",
        "rc",
    )  # hf dataset name and subset
    dataset_size: int = 16
    seed: int = 1337
    batch_size: int = 2


# ## Load model and dataset
def load_model_and_dataset():
    global cfg
    # Load dataset
    print(f"Loading dataset: {cfg.dataset_name}")
    ds = load_dataset(*cfg.dataset_name, split="validation", streaming=False)
    ds = ds.shuffle(seed=cfg.seed).select(range(cfg.dataset_size))
    print(f"Dataset loaded: {len(ds)} examples")

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

    return (
        activations_storage_minus1,
        activations_storage_mid,
        activations_storage_plus1,
    )


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

    prompt_template = """I am going to ask you a question. Answer concisely. End your sentence with {eos_token}. Here
    are some examples of questions that might help you:
    —--
    Question: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
    Answer: Sinclair Lewis{eos_token}
    —--
    Question: {question}
    Answer:
    """
    # DEBUG: print(prompt_template.format(eos_token="<|eos|>", question="Which American-born Sinclair won the Nobel Prize for Literature in 1930?"))
    all_questions = []
    all_ground_truths = []
    all_model_answers = []
    all_activations = []

    for i in tqdm(range(0, len(ds), cfg.batch_size)):
        batch = ds[i : min(i + cfg.batch_size, len(ds))]

        # Prepare questions
        questions = batch["question"]
        all_questions.extend(questions)

        # Store ground truth
        ground_truths = [
            ans["value"][0] if isinstance(ans["value"], list) else ans["value"]
            for ans in batch["answer"]
        ]
        all_ground_truths.extend(ground_truths)

        # Format prompts
        prompts = [
            prompt_template.format(eos_token=tokenizer.eos_token, question=q)
            for q in questions
        ]
        prompts = [[{"role": "user", "content": p}] for p in prompts]

        # Tokenize
        tokenizer.padding_side = "left"  # last token is the prompt's
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokeinze=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # # DEBUG: print proccessed prompt
        # print(f"procced prompts shape: {inputs['input_ids'].shape}")
        # print("attention mask:\n", inputs['attention_mask'])
        # procced_prompts =tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False)
        # print("procced prompts:")
        # for pp in procced_prompts:
        #     print(repr(pp))
        # break

        # Clear previous activations
        activations_storage_minus1.clear()
        activations_storage_mid.clear()
        activations_storage_plus1.clear()

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
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

        # Extract answers (remove prompt)
        prompt_lengths = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_lengths:]
        answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_model_answers.extend(answers)

        # Store activations from the FIRST forward pass (the prompt processing)
        # The first forward pass processes the full prompt and the hooks captures the last token activation
        # The subsequent forward passes proccess the new generated tokens only each time sequence of length 1
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

    # Remove hooks
    cleanup_hooks(model)

    # Concatenate all activations
    activations_tensor = torch.cat(all_activations, dim=0)
    print(f"\nCollected {len(all_questions)} Q&A pairs")
    print(
        f"Activations shape: {activations_tensor.shape} (Mean of middle layers from prompt's last token)"
    )
    return (
        all_questions,
        all_ground_truths,
        all_model_answers,
        activations_tensor,
    )


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
    activations_tensor: torch.Tensor, correctness_array: np.ndarray
) -> np.ndarray:
    print("\nCalculating probe direction...")

    # Convert to numpy for easier computation
    activations_np = activations_tensor.to(torch.float32).numpy()

    # Partition into correct and incorrect
    correct_mask = correctness_array
    incorrect_mask = ~correctness_array

    correct_activations = activations_np[correct_mask]
    incorrect_activations = activations_np[incorrect_mask]

    print(f"Correct activations: {correct_activations.shape}")
    print(f"Incorrect activations: {incorrect_activations.shape}")

    # Calculate centroids
    correct_centroid = correct_activations.mean(axis=0)
    incorrect_centroid = incorrect_activations.mean(axis=0)

    # New origin is the midpoint between centroids
    new_origin = (correct_centroid + incorrect_centroid) / 2

    # Translate to new origin
    correct_centered = correct_activations - new_origin
    incorrect_centered = incorrect_activations - new_origin

    # Calculate mean directions from new origin
    mu_correct = correct_centered.mean(axis=0)
    mu_incorrect = incorrect_centered.mean(axis=0)

    # Direction from incorrect to correct
    direction = 0.5 * (mu_correct - mu_incorrect)

    print(f"\nDirection vector shape: {direction.shape}")
    print(f"Direction vector norm: {np.linalg.norm(direction):.4f}")

    return CorrectnessScorer(direction, new_origin)


# ## Calculate metadata

def calcluate_metadata(scorer, activations_tensor, correctness_array, model):
    global cfg
    print("\nEvaluating scorer on training data...")

    # Convert to numpy for easier computation
    activations_np = activations_tensor.to(torch.float32).numpy()

    # Score all activations
    scores = scorer.score(activations_np)

    # Analyze score distribution
    correct_mask = correctness_array
    incorrect_mask = ~correctness_array
    correct_scores = scores[correct_mask]
    incorrect_scores = scores[incorrect_mask]

    print(f"\nCorrect answers:")
    print(f"  Mean score: {correct_scores.mean():.4f}")
    print(f"  Std score: {correct_scores.std():.4f}")

    print(f"\nIncorrect answers:")
    print(f"  Mean score: {incorrect_scores.mean():.4f}")
    print(f"  Std score: {incorrect_scores.std():.4f}")

    # Calculate separation
    separation = np.abs(correct_scores.mean() - incorrect_scores.mean())
    print(f"\nSeparation (distance between means): {separation:.4f}")

    # Simple threshold classification (at midpoint)
    threshold = (correct_scores.mean() + incorrect_scores.mean()) / 2
    predicted_correct = scores > threshold
    accuracy = (predicted_correct == correctness_array).mean()
    print(f"Training accuracy: {accuracy:.2%}")

    mid_layer_idx = model.config.num_hidden_layers // 2
    layer_indices = [mid_layer_idx - 1, mid_layer_idx, mid_layer_idx + 1]

    metadata = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "dataset_size": cfg.dataset_size,
        "seed": cfg.seed,
        "layers_idx": layer_indices,
        "correct_score_mean": float(correct_scores.mean()),
        "correct_score_std": float(correct_scores.std()),
        "incorrect_score_mean": float(incorrect_scores.mean()),
        "incorrect_score_std": float(incorrect_scores.std()),
        "separation": float(separation),
        "training_accuracy": float(accuracy),
        "threshold (midpoint between means)": float(threshold),
    }
    return metadata


# ## Save

def save(scorer, metadata):
    global cfg
    save_dir = Path(cfg.base_save_dir)
    save_dir = save_dir / "probe_results" / cfg.dataset_name[0] / cfg.model_name
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
    cfg.dataset_size = int(2**12)
    cfg.batch_size = 8
    for model_name in [
        # "Qwen/Qwen2.5-7B-Instruct",
        # "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
    ]:
        print(f"\n\n=== Processing model: {model_name} ===")
        cfg.model_name = model_name
        model, tokenizer, ds = load_model_and_dataset()

        # Prepare activation storage
        activations_storage_minus1: List[torch.Tensor] = []
        activations_storage_mid: List[torch.Tensor] = []
        activations_storage_plus1: List[torch.Tensor] = []

        # Register hooks
        (
            activations_storage_minus1,
            activations_storage_mid,
            activations_storage_plus1,
        ) = register_activation_hooks(
            model,
            activations_storage_minus1,
            activations_storage_mid,
            activations_storage_plus1,
        )

        # Generate and collect activations
        (
            all_questions,
            all_ground_truths,
            all_model_answers,
            activations_tensor,
        ) = generate_and_collect_activations(
            model,
            tokenizer,
            ds,
            activations_storage_minus1,
            activations_storage_mid,
            activations_storage_plus1,
        )

        # Label with judge
        correctness_array = label_with_judge(
            all_questions, all_ground_truths, all_model_answers
        )

        # Calculate correctness direction
        scorer = calculate_correctness_direction(activations_tensor, correctness_array)

        # Calculate metadata
        metadata = calcluate_metadata(
            scorer, activations_tensor, correctness_array, model
        )

        # Save results
        save(scorer, metadata)



if __name__ == "__main__":
    run()




