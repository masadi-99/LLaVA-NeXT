#!/bin/bash
# Example usage of convert_llama_to_evals.py for different modalities and task types

# CMR MCQ (default prediction path as mentioned by user)
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_mcq_cmr_test.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/vqa_cmr_test_grpo/generated_predictions.jsonl \
  --output-json cmr_mcq_output.json \
  --modality cmr \
  --task-type mcq

# CMR VQA
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_vqa_cmr_test.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/vqa_cmr_test_grpo/generated_predictions.jsonl \
  --output-json cmr_vqa_output.json \
  --modality cmr \
  --task-type vqa

# Echo MCQ
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_mcq_echo_test.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/vqa_echo_test_grpo/generated_predictions.jsonl \
  --output-json echo_mcq_output.json \
  --modality echo \
  --task-type mcq

# Echo VQA
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_vqa_echo_test_new.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/vqa_echo_test_grpo/generated_predictions.jsonl \
  --output-json echo_vqa_output.json \
  --modality echo \
  --task-type vqa

# ECG MCQ
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_mcq_ecg_test.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/ecg_vlm_mcq_test_full/generated_predictions.jsonl \
  --output-json ecg_mcq_output.json \
  --modality ecg \
  --task-type mcq

# ECG VQA
python3 convert_llama_to_evals.py \
  --input-json /home/masadi/LLaMA-Factory/data/cardiology_vqa_ecg_test.json \
  --predictions-jsonl /home/masadi/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/vqa_ecg_test/generated_predictions.jsonl \
  --output-json ecg_vqa_output.json \
  --modality ecg \
  --task-type vqa

