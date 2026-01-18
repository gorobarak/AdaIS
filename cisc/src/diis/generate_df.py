from dataclasses import dataclass
from typing import List
from datetime import datetime
import os

from cisc.src import run_lib
from cisc.src.runners import hugging_face_runner
from cisc.src.runners.runner import Runner
from cisc.src.confidence_extraction import (
    AggregatedConfidenceExtractionConfig,
    ConfidenceExtractionType,
)


@dataclass
class Params:
    model_name: str
    num_traces: int = 30  # total number of traces generated per question
    num_rows: int = 128  # number of questions
    max_num_tokens: int = 756
    temp: float = 0.9
    max_workers_stage1: int = 3  # Not used
    max_workers_stage2: int = 120  # Not used
    runner: Runner = None
    tag: str = None
    dataset_names: List[str] = None
    confidence_config: AggregatedConfidenceExtractionConfig = None
    config: run_lib.ExperimentConfiguration = None
    output_base_dir: str = (
        "/home/yandex/APDL2425a/group_12/gorodissky/google-research/cisc/output"
    )

    def __post_init__(self):
        if self.runner is None:
            self.runner = hugging_face_runner.Runner(self.model_name)
        if self.tag is None:
            self.tag = self.model_name.split("/")[-1]
        if self.dataset_names is None:
            self.dataset_names = ["MMLU"]
        if self.confidence_config is None:
            self.confidence_config = AggregatedConfidenceExtractionConfig(
                verbal_confidence=ConfidenceExtractionType.NONE.value,
                confidence_likelihoods=ConfidenceExtractionType.BINARY.value,
                run_sequence_probability=False,
            )
        if self.config is None:
            self.config = run_lib.ExperimentConfiguration(
                num_traces=self.num_traces,
                num_rows=self.num_rows,
                max_num_tokens=self.max_num_tokens,
                temperature=self.temp,
                tag=self.tag,
                confidence_config=self.confidence_config,
            )




for model_name in [
    # "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]:
    
    print("Generatting data for model:", model_name)
    params = Params(model_name=model_name)

    output_base_dir = os.path.join(params.output_base_dir, params.tag)
    output_base_dir_versioned = os.path.join(
        output_base_dir, datetime.now().strftime("%Y_%m_%d_%H:%M")
    )
    all_datasets_results = []

    #  Generate questions and answers
    all_datasets_results.extend(
        run_lib.run_question_answering_on_datasets(
            params.runner,
            params.dataset_names,
            config=params.config,
            max_workers=params.max_workers_stage1,
            output_base_dir=output_base_dir_versioned,
        )
    )

    output_base_dir = os.path.join(output_base_dir, "confidence")
    output_base_dir_versioned = os.path.join(
        output_base_dir, datetime.now().strftime("%Y_%m_%d_%H:%M")
    )
    # Generate confidence scores
    all_datasets_results = run_lib.run_confidence_extraction_on_experiment_results(
        params.runner,
        all_datasets_results,
        config=params.config.confidence_config,
        max_workers=params.max_workers_stage2,
        output_base_dir=output_base_dir_versioned,
    )
    print("=" * 80)
