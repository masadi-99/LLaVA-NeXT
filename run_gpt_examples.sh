#!/bin/bash
# Example usage of run_gpt.py for different modes

# Echo evaluation
python3 run_gpt.py \
  --mode ECHO_IMAGE \
  --data-path echo_mcq_output.json\
  --results-path results/gpt-5_echo_vqa_test.json

# CMR evaluation
python3 run_gpt.py \
  --mode CMR_IMAGE \
  --data-path cmr_mcq_output.json \
  --results-path results/gpt-5_cmr_mcq_test.json

# ECG evaluation
python3 run_gpt.py \
  --mode ECG_IMAGE \
  --data-path ecg_mcq_output.json \
  --results-path results/gpt-5_ecg_mcq_test.json

# Multimodal evaluation
python3 run_gpt.py \
  --mode MULTIMODAL \
  --data-path multimodal_mcq_output.json \
  --results-path results/gpt-5_multimodal_vqa_test.json

