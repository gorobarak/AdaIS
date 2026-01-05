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

"""Extended self-consistency utilities with difficulty scoring."""

from concurrent import futures
import dataclasses
import os
import numpy as np
from cisc.src import self_consistency
from cisc.src.diis.probe import CorrectnessScorer


def results_to_dataframe_with_difficulty(
    self_consistency_results,
):
  """Converts SelfConsistencyResult to DataFrame, including difficulty scores.

  This extends the standard results_to_dataframe function to properly handle
  the difficulty field in traces.

  Args:
    self_consistency_results: A list of `SelfConsistencyResult`.

  Returns:
    Dataframe with a row per trace, including difficulty column.
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
        trace.get("difficulty", None),  # Handle traces with/without difficulty
    ])

  df[[
      "response",
      "exception",
      "answer",
      "verbal_confidence",
      "confidence_likelihoods",
      "response_probability",
      "difficulty",
  ]] = df.traces.apply(flatten_trace)

  normalize_str = lambda s: "" if s is None else re.sub(r"\W", "", s)
  df["is_correct"] = df.apply(
      lambda row: normalize_str(row.answer) == normalize_str(row.golden_label),
      axis=1,
  )
  return df



def run_self_consistency_with_difficulty(
    runner,
    question_id,
    prompt,
    temp,
    num_tokens,
    num_traces,
    dataset,
):
  """Runs self-consistency with difficulty scoring based on embeddings.

  This extends the standard self-consistency algorithm by computing a difficulty
  score for each trace based on the last token's embedding (from the last layer)
  multiplied by a provided difficulty vector.

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
    SelfConsistencyResult with difficulty scores populated in each trace.
  """
  prompts = [prompt] * num_traces
  
  responses = runner.generate(
      prompts, num_tokens, temp, enable_formatting=True, return_embeddings=True
  )

  # Load difficulty scorer
  hf_model_name = runner.hf_model_name
  load_dir = f"/home/yandex/APDL2425a/group_12/gorodissky/google-research/cisc/output/probe_results/mandarjoshi/trivia_qa/{hf_model_name}"
  # get latest checkpoint
  checkpoints = [file for file in os.listdir(load_dir) if file.endswith(".npz")]
  checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(load_dir, x)), reverse=True)
  print("Latest checkpoint:", checkpoints[0])

  scorer = CorrectnessScorer.load(f"{load_dir}/{checkpoints[0]}")
  with futures.ThreadPoolExecutor(len(prompts)) as executor:

    def extract_trace_from_response(response):
      """Extract trace information including difficulty score."""
      # Compute difficulty score from embeddings
      difficulty_score = None
      if response.embeddings is not None:
        embeddings = response.embeddings
        try:
          # Shape: [# layers, # tokens-in-prompt, # embeddings-size]
          num_layers = embeddings.shape[0]
          middle_layers = [int(0.5 * num_layers) + i for i in [-1, 0, 1]]
          last_token_middle_layers_hidden_states = embeddings[middle_layers, -1, :] # shape: [3, embedding_size]
          hidden_state = last_token_middle_layers_hidden_states.mean(axis=0)
          difficulty_score = scorer.score(hidden_state)
        except (IndexError, ValueError, TypeError) as e:
          print(f"Warning: Could not compute difficulty score: {e}")
      
      # Initialize trace with difficulty score
      trace = self_consistency.Trace(
          prompt=response.prompt,
          response=response.response,
          exception=response.exception,
          answer=None,
          answer_span=None,
          difficulty=difficulty_score,
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



