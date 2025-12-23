import json
import os
from openai import AzureOpenAI
import tqdm
import random
import argparse

random.seed(42)

EVAL_SYSTEM_PROMPT_MCQ = """ You are a helpful assistant that evaluates the accuracy of the answers to the questions.
For each question, you will be given the question, the ground truth answer, and the answer provided by a model.
The model sometimes gives an answer in a different format, or slightly different wording. You should consider these as the same answer.
Your task is to evaluate whether the option chosen by the model is the same optionas the ground truth answer.
If the model hasn't provided a specific answer, e.g. "Please provide more images.", "format error", "Unable to determine", etc., that question should be excluded.
Answer strictly in correct JSON format: {"answer": "Correct" if the answer provided by the model is essentially the same as the ground truth answer, "Incorrect" if wrong, "Excluded" if the model hasn't provided a specific answer}
Example: {"answer": "Correct"}
"""

EVAL_SYSTEM_PROMPT_VQA_Old = """ You are a helpful assistant that evaluates the accuracy of the answers to the questions.
For each question, you will be given the question, the ground truth answer, and the answer provided by a model.
Give a likert scale answer to the question of how well the model answers the question.
Give your final likert scale answer and a short explanation for your answer.
Base the evaluation on wether the model's answer is essentially the same as the ground truth answer. Do not consider variations in wording or format. Focus on semantic similarity.
Base your answer on the following scale:
1 = Poor
2 = Average
3 = Good
4 = Very good
5 = Excellent
Answer strictly in correct JSON format: {"answer": "1" if the answer provided by the model is poor, "2" if average, "3" if good, "4" if very good, "5" if excellent, "explanation": "short explanation for your answer"}
Example: {"answer": "4", "explanation": "The model's answer points at the same severity as ground truth answer. However, the model's answer is not as detailed as the ground truth answer."}
Output only in valid JSON format. Only correct JSON format without ```json and ```.
"""

EVAL_SYSTEM_PROMPT_VQA = """You are a helpful assistant that evaluates the accuracy of answers to cardiology questions. For each question, you will be given the question, the ground truth answer, and the answer provided by a model. Evaluate how well the model's answer semantically matches the ground truth, focusing on the correctness of the ground truth's core claims. Ignore differences in wording or format.
Evaluation rules:
Do not penalize the model for providing additional information or details unless they directly contradict the ground truth or cannot be simultaneously true with it. If unsure whether an added detail contradicts the ground truth, assume it is compatible and do not penalize.
Treat extra claims that are not mentioned in the ground truth as neutral. They neither increase nor decrease the score unless they contradict the ground truth.
Break the ground truth into atomic, independent claims. Assess whether the model's answer correctly affirms/denies each of these claims.
Award partial credit when the ground truth contains multiple independent claims and the model correctly addresses some but not all.
Only assign penalties for incorrect statements when they are independent of other claims (i.e., avoid double-penalizing dependent or overlapping errors). If a model statement directly contradicts a ground-truth claim, count it as incorrect for that claim.
Do not use external knowledge to judge extra claims beyond the ground truth; evaluate only relative to the ground truth content and logical compatibility.
Scoring procedure:
Identify the atomic, independent claims in the ground truth.
For each atomic claim, mark the model's answer as: correct (matches/compatible), incorrect (contradicts), or not addressed.
Compute coverage as the fraction of ground-truth claims correctly addressed (correct / total).
Check for any direct contradictions of ground-truth claims. Contradictions lower the score.
Ignore non-contradictory extra details in scoring.

Contradiction detection rules (categorical vs numeric):
If the ground truth is categorical (e.g., normal vs abnormal; present vs absent), treat model statements as contradictory only when they explicitly assert the opposite category (e.g., "dilated," "enlarged," "abnormal," "present," "absent").
Numeric values in the model's answer must not be used to infer a contradiction. Do not apply external thresholds. If the model provides a numerical value without an explicit abnormal label, treat it as neutral or compatible.
Phrases like "within normal limits," "upper limit of normal," "high-normal," "low-normal," or "borderline normal" should be treated as compatible with "normal" unless the model explicitly labels the finding as abnormal (e.g., "dilated," "aneurysmal," "enlarged," "mild dilation").
Only penalize when the model explicitly contradicts the ground truth category. Do not penalize for compatible qualifiers.
Retain the existing rules about breaking ground truth into atomic claims and awarding partial credit only when independent claims are missed or contradicted.
Note that no score should be reduced just because the model has provided additional information, measurements, etc. A score should only be reduced if the model explicitly mentions a phrase that contradicts the ground truth.
Consider "trace" and "mild" as the same severity for grading purposes.

Likert scale mapping:
5 = Excellent: All ground-truth claims are correctly addressed; no contradictions.
4 = Very good: All ground-truth claims are correctly addressed but with minor uncertainty or minor omission of non-critical nuance; no contradictions.
3 = Good: Most ground-truth claims (≥50%) are correctly addressed; no direct contradictions.
2 = Average: Some ground-truth claims (<50%) are correctly addressed, or there is a minor contradiction alongside some correct claims.
1 = Poor: Major contradiction of ground-truth claims or largely incorrect.
Answer strictly in correct JSON format: {"answer": "1" if the answer provided by the model is poor, "2" if average, "3" if good, "4" if very good, "5" if excellent, "explanation": "short explanation for your answer"}
Example of application: Ground truth: "Neither echocardiography nor cardiac MRI shows pericardial effusion. The ECG shows a regular sinus rhythm." Model answer: "No pericardial effusion is present on either echocardiogram or cardiac MRI, and the ECG shows sinus rhythm with occasional premature ventricular complexes." Evaluation: The model correctly addresses both ground-truth claims (no effusion; sinus rhythm). The added PVC detail is extra and compatible with sinus rhythm. Score: {"answer": "5", "explanation": "The model correctly covers all ground-truth claims; the added PVC detail does not contradict the ground truth."} 
"""


# Setup Azure OpenAI client
securegpt_api_key = os.environ.get("SECUREGPT_API_KEY")
headers = {
    'Ocp-Apim-Subscription-Key': securegpt_api_key,
    'Content-Type': 'application/json',
}
API_VERSION = '2024-12-01-preview'
EVAL_MODEL = "gpt-4o-mini"
DEPLOYMENT = EVAL_MODEL
OPENAI_ENDPOINT = os.environ.get('OPENAI_ENDPOINT')
URL = f'https://apim.stanfordhealthcare.org/openai-eastus2/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}'

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=URL,
    azure_deployment=DEPLOYMENT,
    default_headers=headers,
    azure_ad_token=securegpt_api_key,
)


def eval_model(question, gts_answer, model_answer, eval_system_prompt):
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": eval_system_prompt
            },
            {
                "role": "user",
                "content": "Question: " + question + "\n\nGround truth answer: " + gts_answer + "\n\nModel answer: " + model_answer
            }
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model performance on cardiology test data"
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to input JSON file with model results"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output analysis results file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to evaluate (must match field name in input JSON, e.g., 'gpt-5', 'MMC', 'gemini-2.5-pro')"
    )
    parser.add_argument(
        "--question-type",
        required=True,
        choices=["MCQ", "VQA"],
        help="Question type: MCQ (multiple choice) or VQA (visual question answering)"
    )
    
    args = parser.parse_args()
    
    MODEL = args.model
    QUESTION_TYPE = args.question_type
    input_json = args.input_json
    output_path = args.output_path

    print(f"Running analysis with:")
    print(f"  Model: {MODEL}")
    print(f"  Question type: {QUESTION_TYPE}")
    print(f"  Input: {input_json}")
    print(f"  Output: {output_path}")
    print()

    # Load model results
    model_results = json.load(open(input_json))

    # Extract questions and answers based on question type
    if QUESTION_TYPE == "MCQ":
        questions = [result["question"] for result in model_results]
        gts_answers = [result["gts_answer"] for result in model_results]
        model_raw_answers = [result[f"{MODEL}"] for result in model_results]
        EVAL_SYSTEM_PROMPT = EVAL_SYSTEM_PROMPT_MCQ
    elif QUESTION_TYPE == "VQA":
        questions = [result["question"] for result in model_results]
        gts_answers = [result["gts"] for result in model_results]
        model_raw_answers = [result[f"{MODEL}"] for result in model_results]
        EVAL_SYSTEM_PROMPT = EVAL_SYSTEM_PROMPT_VQA
    else:
        raise ValueError(f"Question type {QUESTION_TYPE} not supported")

    if QUESTION_TYPE == "MCQ":
        # Calculate the accuracy
        correct = 0
        excluded = 0
        
        # Store results for each question
        detailed_results = []
        
        for i, (question, gts_answer, model_answer) in enumerate(tqdm.tqdm(
            zip(questions, gts_answers, model_raw_answers), 
            total=len(questions)
        )):
            result = eval_model(question, gts_answer, model_answer, EVAL_SYSTEM_PROMPT)
            print(result)
            result = json.loads(result)
            
            # Create detailed result for this question
            question_result = {
                "question_index": i,
                "question": question,
                "gts_answer": gts_answer,
                "model_answer": model_answer,
                "evaluation": result["answer"]
            }
            detailed_results.append(question_result)
            
            if result["answer"] == "Correct":
                correct += 1
            elif result["answer"] == "Excluded":
                excluded += 1
                print("Excluded question: ", question)
                print("gts_answer: ", gts_answer)
                print("model_answer: ", model_answer)
                print("--------------------------------")
            else:
                print("Incorrect answer for question: ", question)
                print("gts_answer: ", gts_answer)
                print("model_answer: ", model_answer)
                print("--------------------------------")

        # Calculate metrics
        total = len(questions)
        accuracy = correct / (total - excluded) if (total - excluded) > 0 else 0
        
        # Create summary results
        summary = {
            "model": MODEL,
            "question_type": QUESTION_TYPE,
            "total_questions": total,
            "correct": correct,
            "incorrect": total - correct - excluded,
            "excluded": excluded,
            "accuracy": accuracy,
            "accuracy_percentage": accuracy * 100
        }
        
        # Combine summary and detailed results
        final_results = {
            "summary": summary,
            "detailed_results": detailed_results
        }
        
        # Print summary
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct - excluded}")
        print(f"Excluded: {excluded}")
        print(f"Total: {total}")
        print("="*70)
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\n✓ Results saved to: {output_path}")

    elif QUESTION_TYPE == "VQA":
        # Calculate the average likert scale answer and save each question with its likert score
        average_likert_score = 0
        for i in tqdm.tqdm(range(len(model_results))):
            question = model_results[i]["question"]
            gts_answer = model_results[i]["gts"]
            try:
                model_answer = model_results[i][f"{MODEL}_answer"]
            except:
                model_answer = model_results[i][f"{MODEL}"]
            result = eval_model(question, gts_answer, model_answer, EVAL_SYSTEM_PROMPT)
            print(result)
            try:
                result = json.loads(result)
            except:
                result = eval_model(question, gts_answer, model_answer, EVAL_SYSTEM_PROMPT)
                print(result)
                result = json.loads(result)
            average_likert_score += int(result["answer"])
            model_results[i]["likert_score"] = int(result["answer"])
            model_results[i]["likert_score_explanation"] = result["explanation"]
        
        avg_score = average_likert_score / len(model_results)
        
        # Add summary to results
        vqa_results = {
            "summary": {
                "model": MODEL,
                "question_type": QUESTION_TYPE,
                "total_questions": len(model_results),
                "average_likert_score": avg_score
            },
            "detailed_results": model_results
        }
        
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Average likert score: {avg_score:.4f}")
        print(f"Total questions: {len(model_results)}")
        print("="*70)
        
        with open(output_path, "w") as f:
            json.dump(vqa_results, f, indent=4)
        
        print(f"\n✓ Results saved to: {output_path}")

