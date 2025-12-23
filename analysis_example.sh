#!/bin/bash
# Example usage of analysis.py for different models and question types

# MCQ Analysis - GPT-5
python3 analysis.py \
  --input-json results/gpt-5_cmr_mcq_test.json \
  --output-path analysis_results/gpt-5_cmr_mcq_analysis.json \
  --model gpt-5 \
  --question-type MCQ

# MCQ Analysis - MMC
python3 analysis.py \
  --input-json results/MMC_echo_mcq_test.json \
  --output-path analysis_results/MMC_echo_mcq_analysis.json \
  --model MMC \
  --question-type MCQ

# VQA Analysis - GPT-5
python3 analysis.py \
  --input-json results/gpt-5_cmr_vqa_test.json \
  --output-path analysis_results/gpt-5_cmr_vqa_analysis.json \
  --model gpt-5 \
  --question-type VQA

# VQA Analysis - Gemini
python3 analysis.py \
  --input-json results/gemini-2.5-pro_multimodal_vqa_test.json \
  --output-path analysis_results/gemini_multimodal_vqa_analysis.json \
  --model gemini-2.5-pro \
  --question-type VQA

# ECG MCQ Analysis
python3 analysis.py \
  --input-json results/gpt-5_ecg_mcq_test.json \
  --output-path analysis_results/gpt-5_ecg_mcq_analysis.json \
  --model gpt-5 \
  --question-type MCQ

