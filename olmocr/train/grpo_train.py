"""
GRPO (Generative Reward-based Policy Optimization) training script for OlmOCR.
"""

import argparse
import logging
import os
from typing import List, Dict, Any, Optional, Set, Tuple
import asyncio
import json
import random
from pathlib import Path
import glob
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from trl import GRPOConfig, GRPOTrainer
from PIL import Image
import base64
from io import BytesIO

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from olmocr.bench.tests import load_single_test

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class OlmOCRBenchDataset(Dataset):
    """Dataset for loading PDF pages from Olmocr-bench format JSONL files."""
    
    def __init__(
        self,
        bench_data_folder: str,
        processor,
        max_samples: Optional[int] = None,
        target_longest_image_dim: int = 1024,
    ):
        self.bench_data_folder = bench_data_folder
        self.processor = processor
        self.target_longest_image_dim = target_longest_image_dim
        self.max_samples = max_samples
        
        # Find PDF folder
        self.pdf_folder = os.path.join(bench_data_folder, "pdfs")
        if not os.path.exists(self.pdf_folder):
            raise ValueError(f"PDFs folder not found at {self.pdf_folder}")
        
        # Load unique PDFs from JSONL files
        self.samples = self._load_unique_pdfs_from_jsonl()
        
        logger.info(f"Created dataset with {len(self.samples)} unique PDF samples")
    
    def _load_unique_pdfs_from_jsonl(self) -> List[Dict[str, Any]]:
        """Load unique PDFs from JSONL files in the bench_data folder, tracking all test cases per PDF."""
        jsonl_files = sorted(glob.glob(os.path.join(self.bench_data_folder, "*.jsonl")))
        
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in {self.bench_data_folder}")
        
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        # Track unique PDFs and their test cases
        pdf_data: Dict[str, Dict[str, Any]] = {}
        
        for jsonl_file in jsonl_files:
            logger.info(f"Processing {os.path.basename(jsonl_file)}")
            
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        pdf_name = entry.get("pdf")
                        page = entry.get("page", 0)
                        test_id = entry.get("id")
                        
                        if pdf_name and test_id:
                            # Create unique key for PDF+page combination
                            pdf_page_key = f"{pdf_name}::{page}"
                            
                            if pdf_page_key not in pdf_data:
                                # First time seeing this PDF+page
                                pdf_path = os.path.join(self.pdf_folder, pdf_name)
                                pdf_data[pdf_page_key] = {
                                    "pdf_path": pdf_path,
                                    "pdf_name": pdf_name,
                                    "page": page,
                                    "jsonl_file": jsonl_file,
                                    "test_ids": [test_id],
                                    "entries": [entry]
                                }
                            else:
                                # Add test case to existing PDF+page
                                pdf_data[pdf_page_key]["test_ids"].append(test_id)
                                pdf_data[pdf_page_key]["entries"].append(entry)
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing entry in {jsonl_file}: {e}")
                        continue
        
        # Convert to list with sorted keys for reproducibility
        samples = [pdf_data[key] for key in sorted(pdf_data.keys())]
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdf_path = sample["pdf_path"]
        page_num = sample["page"]
        jsonl_file = sample["jsonl_file"]
        test_ids = sample["test_ids"]
        
        try:
            # Render PDF page to base64 image
            image_base64 = render_pdf_to_base64png(
                pdf_path, 
                page_num, 
                target_longest_image_dim=self.target_longest_image_dim
            )
            
            # Convert base64 to PIL Image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Build the text prompt
            text_prompt = build_no_anchoring_v4_yaml_prompt()
            
            # Create messages in the format expected by Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image"},
                    ],
                }
            ]
            
            # Return the required format
            return {
                "prompt": messages,
                "pdf_path": pdf_path,
                "jsonl_file": jsonl_file,
                "test_ids": test_ids,
                "image": image,  # Include the PIL image for processing later
            }
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            # Return None if processing fails
            return None

@lru_cache(maxsize=1024)
def load_specific_tests_cached(jsonl_file: str, test_ids_tuple: tuple):
    """
    Cached version that loads specific tests by their IDs from a JSONL file.
    Uses load_single_test to parse individual test entries.
    
    Args:
        jsonl_file: Path to the JSONL file containing test definitions
        test_ids_tuple: Tuple of test IDs to load (tuple for hashability in lru_cache)
        
    Returns:
        List of test objects matching the specified IDs
    """
    test_ids = set(test_ids_tuple)
 
    relevant_tests = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse just enough to get the ID
                test_data = json.loads(line)
                if test_data.get('id') in test_ids:
                    # Use load_single_test to properly parse and validate the test
                    test = load_single_test(test_data)
                    relevant_tests.append(test)
                    # Early exit if we've found all tests
                    if len(relevant_tests) == len(test_ids):
                        break
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error parsing test line: {e}")
                continue
    
    return relevant_tests


def evaluate_single_completion(args: Tuple[int, Any, str, str, List[str]]) -> Tuple[int, Optional[float]]:
    """
    Helper function to evaluate a single completion against its tests.
    
    Args:
        args: Tuple of (index, completion, jsonl_file, pdf_path, test_ids)
        
    Returns:
        Tuple of (index, reward) where reward is float or None for errors
    """
    i, completion, comp_jsonl_file, comp_pdf_path, comp_test_ids = args
    
    logger.info(f"Completion {i}: PDF: {comp_pdf_path}, JSONL: {comp_jsonl_file}, Test IDs: {comp_test_ids}")
    
    if completion is None or not (isinstance(completion, str) or isinstance(completion, list)):
        logger.warning(f"Invalid completion at index {i}: {type(completion)}")
        logger.warning(f"completion: {completion}")
        return i, None
    
    if comp_jsonl_file is None or comp_test_ids is None or len(comp_test_ids) == 0:
        logger.warning(f"Missing metadata for completion {i}")
        return i, None

    if isinstance(completion, list):
        completion = completion[0]["content"]
    
    try:
        # Load only the specific tests we need from the JSONL file (cached)
        # Convert list to tuple for hashability in lru_cache
        relevant_tests = load_specific_tests_cached(comp_jsonl_file, tuple(comp_test_ids))
        
        if not relevant_tests:
            logger.warning(f"No relevant tests found for test IDs: {comp_test_ids}")
            return i, None
        
        logger.info(f"Found {len(relevant_tests)} relevant tests for completion {i}")
        
        # Run all relevant tests on this completion
        passed = 0
        total = len(relevant_tests)
        
        for test in relevant_tests:
            try:
                test_passed, failure_reason = test.run(completion)
                if test_passed:
                    passed += 1
                else:
                    logger.debug(f"Test {test.id} failed: {failure_reason}")
            except Exception as e:
                logger.warning(f"Error running test {test.id}: {e}")
                # Count errored tests as failures
                continue
        
        # Calculate reward as proportion of tests passed
        reward = passed / total if total > 0 else 0.0
        
        logger.info(f"Completion {i}: {passed}/{total} tests passed, reward={reward:.3f}")
        return i, reward
        
    except Exception as e:
        logger.error(f"Error processing completion {i}: {e}")
        return i, None


def olmocr_bench_reward(prompts, completions: list[str] | list[list[dict]], completion_ids: list[list[int]], pdf_path: list[str], jsonl_file: list[str], test_ids: list[list[str]], **kwargs):
    """
    Reward function that runs unit tests on completions and returns average pass rate.
    Uses ThreadPoolExecutor to evaluate completions in parallel.
    
    For each completion, loads the corresponding tests from the JSONL file and runs them.
    Returns the proportion of tests that pass as the reward score.
    
    Args:
        prompts: List of prompts
        completions: List of generated completions (model outputs)
        completion_ids: List of completion token IDs
        pdf_path: List of PDF file paths (one per completion)
        jsonl_file: List of JSONL file paths containing test definitions (one per completion)
        test_ids: List of test ID lists associated with each PDF page (one list per completion)
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (float) based on test pass rates, or None for errors
    """
    logger.info(f"Running olmocr bench reward function for {len(completions)} completions")
    
    # Prepare arguments for parallel processing
    eval_args = []
    for i, completion in enumerate(completions):
        comp_pdf_path = pdf_path[i] if i < len(pdf_path) else None
        comp_jsonl_file = jsonl_file[i] if i < len(jsonl_file) else None
        comp_test_ids = test_ids[i] if i < len(test_ids) else []
        eval_args.append((i, completion, comp_jsonl_file, comp_pdf_path, comp_test_ids))
    
    # Process completions in parallel using ThreadPoolExecutor
    rewards = [None] * len(completions)  # Pre-allocate results list
    
    # Use number of CPUs for thread pool size, with a reasonable maximum
    max_workers = min(os.cpu_count() or 4, 16, len(completions))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once
        futures = [executor.submit(evaluate_single_completion, args) for args in eval_args]
        
        # Collect results as they complete (but maintain order)
        for future in futures:
            idx, reward = future.result()
            rewards[idx] = reward
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO training for OlmOCR")
    parser.add_argument(
        "--train_bench_data_folder", 
        type=str, 
        required=True,
        help="Path to training bench data folder containing JSONL files and pdfs subfolder"
    )
    parser.add_argument(
        "--eval_bench_data_folder", 
        type=str, 
        required=False,
        default=None,
        help="Path to evaluation bench data folder (optional, uses train folder if not specified)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model checkpoint to load"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/grpo_test",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (default: use all)"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=10,
        help="Maximum number of evaluation samples to use (default: 10)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="olmocr-grpo",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (default: auto-generated)"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bnpo",
        choices=["bnpo", "grpo", "exo"],
        help="Loss formulation to use (default: bnpo)"
    )
    parser.add_argument(
        "--scale_rewards",
        action="store_true",
        default=True,
        help="Whether to scale rewards by their standard deviation (default: True)"
    )
    parser.add_argument(
        "--no_scale_rewards",
        action="store_false",
        dest="scale_rewards",
        help="Disable reward scaling"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient for reference model (default: 0.0, no reference model)"
    )
    parser.add_argument(
        "--importance_sampling_level",
        type=str,
        default="token",
        choices=["token", "sequence"],
        help="Level for importance sampling ratios (default: token)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    logger.info(f"Initialized wandb project: {args.wandb_project}")
    report_to = ["wandb"]

    
    # Verify train bench_data_folder exists
    if not os.path.exists(args.train_bench_data_folder):
        logger.error(f"Train bench data folder not found: {args.train_bench_data_folder}")
        return
    
    # Set eval folder to train folder if not specified
    if args.eval_bench_data_folder is None:
        args.eval_bench_data_folder = args.train_bench_data_folder
        logger.info(f"Using train folder for evaluation: {args.eval_bench_data_folder}")
    elif not os.path.exists(args.eval_bench_data_folder):
        logger.error(f"Eval bench data folder not found: {args.eval_bench_data_folder}")
        return
    
    # Load processor
    logger.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    if "Qwen2-VL" in args.model_name:
        model_class = Qwen2VLForConditionalGeneration
    else:
       model_class = Qwen2_5_VLForConditionalGeneration
    
    model = model_class.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Create training dataset
    logger.info(f"Creating training dataset from: {args.train_bench_data_folder}")
    train_dataset = OlmOCRBenchDataset(
        bench_data_folder=args.train_bench_data_folder,
        processor=processor,
        max_samples=args.max_train_samples,
        target_longest_image_dim=1288,
    )
    
    if len(train_dataset) == 0:
        logger.error("No samples found in training dataset!")
        return
    
    # Create evaluation dataset
    logger.info(f"Creating evaluation dataset from: {args.eval_bench_data_folder}")
    eval_dataset = OlmOCRBenchDataset(
        bench_data_folder=args.eval_bench_data_folder,
        processor=processor,
        max_samples=args.max_eval_samples,
        target_longest_image_dim=1288,
    )
    
    if len(eval_dataset) == 0:
        logger.warning("No samples found in evaluation dataset, using training dataset for eval")
        eval_dataset = train_dataset
    
    # Set up GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_steps=10,
        max_prompt_length=3000,
        max_completion_length=3000,
        temperature=0.7,
        report_to=report_to,
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=0,
        
        # GRPO-specific parameters
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        beta=args.beta,
        importance_sampling_level=args.importance_sampling_level,

        # Vllm setup to speed up generation
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        log_completions=True,
    )
    
    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=olmocr_bench_reward,
    )
    
    # Start training
    logger.info("Starting GRPO training")
    try:
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        logger.info("Training completed successfully!")
        
        # Close wandb if it was used
        if args.use_wandb:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()