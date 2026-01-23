# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extended self-consistency utilities with correctness scoring."""

from concurrent import futures
import dataclasses
import os
import numpy as np
import torch
from adais import self_consistency
from adais.adaptive.probe import CorrectnessScorer


def results_to_dataframe_with_correctness_score(
    self_consistency_results,
):
  """Converts SelfConsistencyResult to DataFrame, including correctness scores.

  This extends the standard results_to_dataframe function to properly handle
  the correctness_score field in traces.

  Args:
    self_consistency_results: A list of `SelfConsistencyResult`.

  Returns:
    Dataframe with a row per trace, including correctness_score column.
  """
  import pandas as pd
  import re
  
  df = pd.DataFrame(self_consistency_results)
  # Create row per trace and check each trace's answer.
  df = df.explode("traces").reset_index()

  # Flatten the trace into columns so it would be easier to debug.
  def flatten_trace(trace):
    """trace is a dict representation of a `SelfConsistencyResult.Trace`."""
    has_conf = pd.notna(trace["confidence"])
    return pd.Series([
        trace["response"],
        trace["exception"],
        trace["answer"],
        trace["confidence"]["verbal_conf"] if has_conf else None,
        trace["confidence"]["confidence_likelihoods"] if has_conf else None,
        trace["confidence"]["response_probability"] if has_conf else None,
        trace["correctness_score"]
    ])

  df[[
      "response",
      "exception",
      "answer",
      "verbal_confidence",
      "confidence_likelihoods",
      "response_probability",
      "correctness_score",
  ]] = df.traces.apply(flatten_trace)
  def normalize_str(s):
    if pd.isna(s):
      return ""
    return re.sub(r"\W", "", str(s))
  
  df["is_correct"] = df.apply(
      lambda row: normalize_str(row["answer"]) == normalize_str(row["golden_label"]),
      axis=1,
  )
  return df



def run_self_consistency_with_correctness_score(
    runner,
    question_id,
    prompt,
    temp,
    num_tokens,
    num_traces,
    dataset,
):
  """Runs self-consistency with correctness scoring based on embeddings.

  This extends the standard self-consistency algorithm by computing a correctness
  score for each trace based on the last prompt's token embedding (from the middle layers)
  multiplied by a preprocessed correctness scorer.

  Args:
    runner: the runner to use for querying the model.
    question_id: the question id.
    prompt: the prompt to use for querying the model.
    temp: the temperature to use for querying the model.
    num_tokens: the number of tokens to use for querying the model.
    num_traces: the number of traces to run. Each would be a different call to
      the model.
    dataset: the dataset which includes the instructions on how to format the
      prompts and how to extract the answers.

  Returns:
    SelfConsistencyResult with correctness scores populated in each trace.
  """
  prompts = [prompt] * num_traces
  
  responses = runner.generate(
      prompts, num_tokens, temp, enable_formatting=True, return_embeddings=True
  )

  # Load correctness scorer (latest checkpoint)
  hf_model_name = runner.hf_model_name
  load_dir = f"/home/yandex/APDL2425a/group_12/gorodissky/AdaIS/output/probe_results/MMLU/{hf_model_name}"
  checkpoints = [file for file in os.listdir(load_dir) if file.endswith(".npz")]
  checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(load_dir, x)), reverse=True)
  scorer = CorrectnessScorer.load(f"{load_dir}/{checkpoints[0]}")
  assert len(scorer.layer_indices.shape) == 1, "Expected 1D array of layer indices"
  layer_indices = scorer.layer_indices.tolist()
  print("loaded:", f"probe_results/MMLU/{hf_model_name}/{checkpoints[0]}")
  
  # Extract traces from responses in parallel
  with futures.ThreadPoolExecutor(len(prompts)) as executor:

    def extract_trace_from_response(response):
      """Extract trace information including correctness score."""
      
      # Compute correctness score from embeddings
      correctness_score = None
      if response.embeddings is not None:
        embeddings = response.embeddings
        try:
          # Shape: [# layers, # tokens-in-prompt, # embeddings-size]
          last_token_middle_layers_hidden_states = embeddings[layer_indices, -1, :].to(torch.float32) # shape: [3, embedding_size]
          hidden_state = last_token_middle_layers_hidden_states.mean(dim=0, keepdim=True)
          correctness_score = scorer.score(hidden_state)
        except Exception as e:
          print("Warning: Could not compute correctness score")
          raise e

      
      # Initialize trace with correctness score
      trace = self_consistency.Trace(
          prompt=response.prompt,
          response=response.response,
          exception=response.exception,
          answer=None,
          answer_span=None,
          correctness_score=correctness_score,
      )
      
      response_text = response.response
      if response_text is None:
        return trace
      
      # Extract answer from response
      ans, span = dataset.extract_answer(response_text)
      if (not ans) or (not span):
        return trace
      
      return dataclasses.replace(trace, answer=ans, answer_span=span)

    traces = list(executor.map(extract_trace_from_response, responses))

  return self_consistency.SelfConsistencyResult(
      question_id=question_id,
      prompt=prompt,
      temperature=temp,
      traces=traces,
  )



