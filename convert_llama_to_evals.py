#!/usr/bin/env python3
"""
Convert LLaMA-Factory data format to Evals format.

This script converts cardiology test data and model predictions into the Evals format.
Supports both MCQ and VQA types, and CMR/Echo (videos) and ECG (images).
"""

import json
import argparse
import sys
from pathlib import Path


def parse_gpt_answer(gpt_value):
    """
    Parse GPT answer to extract reasoning and final answer.
    
    Args:
        gpt_value: String containing <think>reasoning</think>\nAnswer format
        
    Returns:
        tuple: (reasoning, answer) where reasoning includes <think> tags
    """
    if "<think>" in gpt_value and "</think>" in gpt_value:
        # Find the <think>...</think> part
        think_start = gpt_value.find("<think>")
        think_end = gpt_value.find("</think>") + len("</think>")
        reasoning = gpt_value[think_start:think_end]
        # Everything after </think> (stripping leading newlines/spaces)
        answer = gpt_value[think_end:].strip()
        return reasoning, answer
    else:
        # No reasoning, entire thing is the answer
        return "", gpt_value.strip()


def convert_data(input_json_path, predictions_jsonl_path, output_json_path, modality, task_type):
    """
    Convert LLaMA-Factory format to Evals format.
    
    Args:
        input_json_path: Path to cardiology_*_test.json
        predictions_jsonl_path: Path to generated_predictions.jsonl
        output_json_path: Path to output file
        modality: One of 'cmr', 'echo', 'ecg'
        task_type: One of 'mcq', 'vqa'
    """
    # Load input data
    print(f"Loading input data from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Load predictions
    print(f"Loading predictions from {predictions_jsonl_path}...")
    predictions = []
    with open(predictions_jsonl_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Check if counts match
    if len(input_data) != len(predictions):
        raise ValueError(
            f"Mismatch in data counts: {len(input_data)} items in input JSON "
            f"but {len(predictions)} predictions in JSONL"
        )
    
    print(f"Converting {len(input_data)} items...")
    
    # Determine field names based on modality
    media_field = "images" if modality == "ecg" else "videos"
    
    # Convert each item
    output_data = []
    for idx, (item, pred) in enumerate(zip(input_data, predictions)):
        # Get conversations
        conversations = item["conversations"]
        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"]
        
        # Extract question (remove <video> or <image> tags)
        question = human_msg.replace("<video>", "").replace("<image>", "").strip()
        
        # Parse ground truth
        gts = gpt_msg
        if task_type == "mcq":
            gts_reasoning, gts_answer = parse_gpt_answer(gpt_msg)
        
        # Get model prediction
        mmc = pred["predict"]
        
        # Build output item
        output_item = {
            "id": item["id"],
            "modality": modality.upper(),
            "type": "Multiple-choice" if task_type == "mcq" else "Open-ended",
            "question": question,
            media_field: item.get(media_field, item.get("videos" if media_field == "images" else "images")),
            "gts": gts,
        }
        
        # Add MCQ-specific fields
        if task_type == "mcq":
            output_item["gts_reasoning"] = gts_reasoning
            output_item["gts_answer"] = gts_answer
        
        # Add model prediction
        output_item["MMC"] = mmc
        
        # Add metadata
        output_item["metadata"] = {}
        
        output_data.append(output_item)
    
    # Write output
    print(f"Writing output to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully converted {len(output_data)} items")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaMA-Factory data format to Evals format"
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to input JSON file (e.g., cardiology_mcq_cmr_test.json)"
    )
    parser.add_argument(
        "--predictions-jsonl",
        required=True,
        help="Path to predictions JSONL file (e.g., generated_predictions.jsonl)"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["cmr", "echo", "ecg"],
        help="Modality type (cmr/echo have videos, ecg has images)"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["mcq", "vqa"],
        help="Task type (mcq has reasoning/answer fields, vqa does not)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_data(
            args.input_json,
            args.predictions_jsonl,
            args.output_json,
            args.modality,
            args.task_type
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

