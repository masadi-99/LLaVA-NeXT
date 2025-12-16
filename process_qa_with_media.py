#!/usr/bin/env python3
"""
Process QA datasets and create LLaMA-Factory format with file existence checks.

This script takes generated QA JSON files and converts them to LLaMA-Factory format,
checking that the corresponding video/image files exist.
"""

import json
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def get_study_id_from_csv(csv_df: pd.DataFrame, report_id: int, modality: str) -> str:
    """
    Get study ID from CSV based on report_id and modality.
    
    Args:
        csv_df: DataFrame with dropped NA values
        report_id: Index in the dataframe
        modality: One of 'echo', 'cmr', 'ecg'
        
    Returns:
        Study ID string or None if not found
    """
    if report_id >= len(csv_df):
        return None
    
    row = csv_df.iloc[report_id]
    
    if modality in ['echo', 'echo_new']:
        # Echo uses person_source_value and study_source_value
        person_id = row.get('person_source_value', '')
        study_id = row.get('study_source_value', '')
        if person_id and study_id:
            return f"{person_id}-{study_id}"
        return None
    elif modality == 'cmr':
        # CMR uses person_source_value
        return row.get('person_source_value', None)
    elif modality == 'ecg':
        # ECG uses deid_filename
        return row.get('deid_filename', None)
    else:
        raise ValueError(f"Unknown modality: {modality}")


def construct_file_path(study_id: str, modality: str, root_path: str) -> str:
    """
    Construct the expected file path for a given study_id and modality.
    
    Args:
        study_id: Study identifier
        modality: One of 'echo', 'cmr', 'ecg'
        root_path: Root directory for media files
        
    Returns:
        Full file path (absolute)
    """
    # Convert to absolute path
    root_path = os.path.abspath(root_path)
    
    if modality in ['echo', 'echo_new']:
        # Echo videos: {root_path}/{study_id}_grid_small.mp4
        return os.path.join(root_path, f"{study_id}_grid_small.mp4")
    elif modality == 'cmr':
        # CMR videos: {root_path}/{study_id1}-{study_id2}_grid.mp4
        # The CMR videos have varied naming, need to search for matching files
        # For now, we'll try to find files that start with the person_source_value
        return os.path.join(root_path, study_id)
    elif modality == 'ecg':
        # ECG images: {root_path}/{study_id}.png
        return os.path.join(root_path, f"{study_id}.png")
    else:
        raise ValueError(f"Unknown modality: {modality}")


def find_cmr_video(study_id: str, root_path: str) -> str:
    """
    Find CMR video file that matches the study_id pattern.
    CMR videos have format like: EA64637a5-EA6463b89_grid.mp4
    where the first part (person_source_value) should match study_id.
    
    Args:
        study_id: person_source_value to match
        root_path: Root directory for CMR videos
        
    Returns:
        Full path to matching video (absolute) or None
    """
    # Convert to absolute path
    root_path = os.path.abspath(root_path)
    
    if not os.path.exists(root_path):
        return None
    
    # Look for files that start with study_id
    for filename in os.listdir(root_path):
        if filename.startswith(study_id) and filename.endswith('_grid.mp4'):
            return os.path.join(root_path, filename)
    
    return None


def check_file_exists(study_id: str, modality: str, root_path: str) -> str:
    """
    Check if the media file exists for a given study.
    
    Args:
        study_id: Study identifier
        modality: One of 'echo', 'cmr', 'ecg'
        root_path: Root directory for media files
        
    Returns:
        Full path if file exists, None otherwise
    """
    if modality == 'cmr':
        # Special handling for CMR videos
        return find_cmr_video(study_id, root_path)
    else:
        file_path = construct_file_path(study_id, modality, root_path)
        if os.path.exists(file_path):
            return file_path
    
    return None


def create_llama_factory_entry_mcq(
    qa_item: Dict[str, Any],
    study_id: str,
    file_path: str,
    modality: str,
    qa_index: int
) -> Dict[str, Any]:
    """
    Create a LLaMA-Factory format entry for MCQ.
    
    Args:
        qa_item: Question item from the generated JSON
        study_id: Study identifier
        file_path: Path to the media file
        modality: One of 'echo', 'cmr', 'ecg'
        qa_index: Index of the question for this report
        
    Returns:
        Dictionary in LLaMA-Factory format
    """
    is_variant = qa_item.get('is_none_variant', False)
    variant_suffix = '_variant' if is_variant else ''
    
    # Format question with options
    question_text = qa_item['question']
    options = qa_item['options']
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
    
    formatted_question = question_text + '\n'
    for label, option in zip(option_labels, options):
        formatted_question += f"{label}. {option}\n"
    formatted_question = formatted_question.rstrip()
    
    # Add media tag
    if modality == 'ecg':
        formatted_question = '<image>' + formatted_question
        media_key = 'images'
    else:
        formatted_question = '<video>' + formatted_question
        media_key = 'videos'
    
    # Format answer with thinking and final answer
    correct_answer = qa_item['correct_answer']
    explanation = qa_item.get('explanation', '')
    
    if explanation:
        formatted_answer = f"<think>{explanation}</think>\n{correct_answer}"
    else:
        formatted_answer = correct_answer
    
    entry = {
        'id': f"{study_id}_mcq_{qa_index}{variant_suffix}",
        'study_id': study_id.split('-')[0] if '-' in study_id else study_id,  # Keep only person_source_value
        'conversations': [
            {
                'from': 'human',
                'value': formatted_question
            },
            {
                'from': 'gpt',
                'value': formatted_answer
            }
        ],
        media_key: [file_path]
    }
    
    return entry


def create_llama_factory_entry_vqa(
    qa_item: Dict[str, Any],
    study_id: str,
    file_path: str,
    modality: str,
    qa_index: int
) -> Dict[str, Any]:
    """
    Create a LLaMA-Factory format entry for VQA.
    
    Args:
        qa_item: Question item from the generated JSON
        study_id: Study identifier
        file_path: Path to the media file
        modality: One of 'echo', 'cmr', 'ecg'
        qa_index: Index of the question for this report
        
    Returns:
        Dictionary in LLaMA-Factory format
    """
    # Format question
    question_text = qa_item['question']
    
    # Add media tag
    if modality == 'ecg':
        formatted_question = '<image>' + question_text
        media_key = 'images'
    else:
        formatted_question = '<video>' + question_text
        media_key = 'videos'
    
    # Format answer
    answer_text = qa_item['answer']
    
    entry = {
        'id': f"{study_id}_vqa_{qa_index}",
        'study_id': study_id.split('-')[0] if '-' in study_id else study_id,  # Keep only person_source_value
        'conversations': [
            {
                'from': 'human',
                'value': formatted_question
            },
            {
                'from': 'gpt',
                'value': answer_text
            }
        ],
        media_key: [file_path]
    }
    
    return entry


def process_qa_dataset(
    questions_json_path: str,
    csv_path: str,
    modality: str,
    question_type: str,
    root_path: str,
    output_json_path: str
) -> None:
    """
    Process QA dataset and create LLaMA-Factory format output.
    
    Args:
        questions_json_path: Path to the generated questions JSON
        csv_path: Path to the CSV file for backtrackin study IDs
        modality: One of 'echo', 'cmr', 'ecg'
        question_type: One of 'mcq', 'vqa'
        root_path: Root directory for media files
        output_json_path: Path to save the output JSON
    """
    print(f"\n{'='*80}")
    print(f"Processing QA Dataset")
    print(f"{'='*80}")
    print(f"Modality: {modality}")
    print(f"Question Type: {question_type}")
    print(f"Questions JSON: {questions_json_path}")
    print(f"CSV: {csv_path}")
    print(f"Media Root: {root_path}")
    print(f"Output: {output_json_path}")
    print(f"{'='*80}\n")
    
    # Load questions JSON
    print("Loading questions JSON...")
    with open(questions_json_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    print(f"✓ Loaded {len(questions_data)} reports with questions")
    
    # Load CSV and apply dropna()
    print(f"\nLoading CSV from {csv_path}...")
    csv_df = pd.read_csv(csv_path)
    print(f"✓ Loaded CSV with {len(csv_df)} rows")
    
    # Apply dropna() based on modality
    if modality in ['echo', 'echo_new']:
        csv_df = csv_df.dropna(subset=['note_text'])
    elif modality == 'cmr':
        csv_df = csv_df.dropna(subset=['note_text'])
    elif modality == 'ecg':
        csv_df = csv_df.dropna(subset=['ECG_INTERP'])
    
    print(f"✓ After dropna(): {len(csv_df)} rows")
    
    # Process each report
    output_entries = []
    total_questions = 0
    missing_files = 0
    processed_questions = 0
    
    print(f"\nProcessing questions...")
    for report_data in questions_data:
        report_id = report_data['report_id']
        questions = report_data.get('questions', [])
        
        if not questions:
            continue
        
        # Get study ID from CSV
        study_id = get_study_id_from_csv(csv_df, report_id, modality)
        if not study_id:
            print(f"⚠ Warning: Could not find study ID for report_id {report_id}")
            missing_files += len(questions)
            continue
        
        # Check if media file exists
        file_path = check_file_exists(study_id, modality, root_path)
        if not file_path:
            print(f"⚠ Warning: Media file not found for study_id {study_id} (report_id {report_id})")
            missing_files += len(questions)
            continue
        
        # Process each question for this report
        for qa_index, qa_item in enumerate(questions):
            total_questions += 1
            
            try:
                if question_type == 'mcq':
                    entry = create_llama_factory_entry_mcq(
                        qa_item, study_id, file_path, modality, qa_index
                    )
                elif question_type == 'vqa':
                    entry = create_llama_factory_entry_vqa(
                        qa_item, study_id, file_path, modality, qa_index
                    )
                else:
                    raise ValueError(f"Unknown question type: {question_type}")
                
                output_entries.append(entry)
                processed_questions += 1
                
            except Exception as e:
                print(f"⚠ Warning: Error processing question {qa_index} for report_id {report_id}: {e}")
                continue
    
    # Save output
    print(f"\nSaving output to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Processing Complete")
    print(f"{'='*80}")
    print(f"Total questions in input: {total_questions}")
    print(f"Successfully processed: {processed_questions}")
    print(f"Missing files: {missing_files}")
    print(f"Output entries: {len(output_entries)}")
    print(f"✓ Output saved to: {output_json_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Process QA datasets and create LLaMA-Factory format with file existence checks'
    )
    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        choices=['echo', 'echo_new', 'cmr', 'ecg'],
        help='Modality: echo, echo_new, cmr, or ecg'
    )
    parser.add_argument(
        '--question-type',
        type=str,
        required=True,
        choices=['mcq', 'vqa'],
        help='Question type: mcq or vqa'
    )
    parser.add_argument(
        '--questions-json',
        type=str,
        required=True,
        help='Path to the questions JSON file (e.g., cardiology_mcq_dataset_echo.json)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the CSV file for backtrackin study IDs'
    )
    parser.add_argument(
        '--root-path',
        type=str,
        required=True,
        help='Root directory for video/image files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the output JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.questions_json):
        print(f"Error: Questions JSON file not found: {args.questions_json}")
        return
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.root_path):
        print(f"Error: Root path not found: {args.root_path}")
        return
    
    # Process the dataset
    process_qa_dataset(
        questions_json_path=args.questions_json,
        csv_path=args.csv,
        modality=args.modality,
        question_type=args.question_type,
        root_path=args.root_path,
        output_json_path=args.output
    )


if __name__ == '__main__':
    main()

