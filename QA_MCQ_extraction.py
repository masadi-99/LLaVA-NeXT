# pip install openai>=1.30.0
import os
import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import copy

import pandas as pd

# -----------------------------
# vLLM / OpenAI-compatible client
# -----------------------------
API_BASE = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("LV_FACTS_MODEL", "openai/gpt-oss-120b")
client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)

# -----------------------------
# Modality-specific diagnostic categories
# -----------------------------
DIAGNOSTIC_CATEGORIES = {
    "ecg": """
COMMON ECG DIAGNOSES (use as reference for plausible distractors):
- Rhythm: Sinus rhythm, atrial fibrillation, atrial flutter, supraventricular tachycardia, ventricular tachycardia, junctional rhythm, atrial tachycardia, multifocal atrial tachycardia
- Conduction abnormalities: First-degree AV block, second-degree AV block (Mobitz I/II), third-degree AV block, right bundle branch block, left bundle branch block, bifascicular block, trifascicular block
- Axis deviation: Left axis deviation, right axis deviation
- Hypertrophy: Left ventricular hypertrophy, right ventricular hypertrophy, left atrial enlargement, right atrial enlargement
- Ischemia/Infarction: Acute ST-elevation MI (anterior/inferior/lateral/posterior), non-ST-elevation MI, acute coronary syndrome, old myocardial infarction
- Arrhythmias: Premature atrial contractions, premature ventricular contractions, bigeminy, trigeminy
- Patterns: Brugada pattern, early repolarization, pericarditis, digitalis effect, Wellens syndrome, Takotsubo pattern
- Other: Pre-excitation (WPW), prolonged QT interval, short QT interval, low voltage
""",
    "echo": """
COMMON ECHO DIAGNOSES (use as reference for plausible distractors):
- Cardiomyopathy: Dilated cardiomyopathy, hypertrophic cardiomyopathy, restrictive cardiomyopathy, Takotsubo cardiomyopathy, arrhythmogenic right ventricular cardiomyopathy
- Systolic dysfunction: Mild/moderate/severe LV systolic dysfunction, RV systolic dysfunction
- Diastolic dysfunction: Grade I/II/III diastolic dysfunction, restrictive filling pattern
- Valve disease: Aortic stenosis (mild/moderate/severe), aortic regurgitation, mitral stenosis, mitral regurgitation, tricuspid regurgitation, pulmonary regurgitation
- Structural: Atrial septal defect, ventricular septal defect, patent foramen ovale, patent ductus arteriosus
- Masses: Left atrial thrombus, left ventricular thrombus, atrial myxoma, vegetation, tumor
- Wall motion: Regional wall motion abnormality, apical ballooning, basal/mid/apical hypokinesis
- Pericardial: Pericardial effusion (small/moderate/large), cardiac tamponade, pericardial constriction
- Aortic: Aortic aneurysm, aortic dissection, dilated aortic root
- Prosthetic: Prosthetic valve dysfunction, paravalvular leak, valve thrombosis
- Pulmonary hypertension: Mild/moderate/severe pulmonary hypertension
""",
    "cmr": """
COMMON CMR DIAGNOSES (use as reference for plausible distractors):
- Cardiomyopathy: Dilated cardiomyopathy, hypertrophic cardiomyopathy, restrictive cardiomyopathy, arrhythmogenic right ventricular cardiomyopathy, non-compaction cardiomyopathy, Takotsubo cardiomyopathy
- Ischemic disease: Acute myocardial infarction, chronic myocardial infarction, myocardial ischemia, viable myocardium, non-viable myocardium
- Inflammation: Acute myocarditis, chronic myocarditis, pericarditis, cardiac sarcoidosis
- Infiltrative: Cardiac amyloidosis, cardiac sarcoidosis, hemochromatosis, Fabry disease
- Structural: Atrial septal defect, ventricular septal defect, patent ductus arteriosus, coarctation of aorta, anomalous coronary artery
- Masses: Left ventricular thrombus, atrial thrombus, cardiac fibroma, cardiac lipoma, metastasis
- Pericardial: Pericardial effusion, constrictive pericarditis, pericardial cyst
- Valvular: Aortic stenosis, aortic regurgitation, mitral regurgitation
- Aortic: Aortic aneurysm, aortic dissection, dilated aortic root
- Other: Pulmonary hypertension, RV dysfunction, LV aneurysm, ventricular non-compaction
"""
}

DIAGNOSTIC_CATEGORIES["echo_new"] = DIAGNOSTIC_CATEGORIES["echo"]

# -----------------------------
# System prompt for MCQ generation
# -----------------------------
def create_mcq_system_prompt(modality: str) -> str:
    """
    Generate modality-specific system prompt for multiple-choice diagnostic questions
    
    Args:
        modality: One of 'ecg', 'echo', or 'cmr'
        
    Returns:
        System prompt string
    """
    if modality not in DIAGNOSTIC_CATEGORIES:
        raise ValueError(f"Invalid modality: {modality}. Must be one of: {list(DIAGNOSTIC_CATEGORIES.keys())}")
    
    diagnostic_info = DIAGNOSTIC_CATEGORIES[modality]
    
    return f"""You are an expert in cardiology imaging ({modality.upper()}). Your task is to generate multiple-choice diagnostic questions based on imaging reports.

CRITICAL REQUIREMENTS:
1. Generate AS MANY diagnostic multiple-choice questions as possible from the report (if none are present, return an empty list of questions)
2. Each question should focus on a SPECIFIC diagnostic finding visible in the images
3. Questions must be answerable by looking at the images alone
4. Each question must have EXACTLY ONE correct answer
5. All incorrect options (distractors) must be:
   - Plausible alternative diagnoses that would be EXPLICITLY mentioned in the report if they were present
   - Clinically significant findings that wouldn't coexist with the correct answer
   - Mutually exclusive with the correct diagnosis
6. Use 4-5 options per question (1 correct + 3-4 distractors)
7. Use present tense and direct observational language
8. NEVER use phrases like: "mentioned", "not mentioned", "reported", "according to the report"

{diagnostic_info}

QUESTION STRUCTURE:
- Frame questions as: "What [diagnostic finding] is present?" or "What is the diagnosis for [structure/region]?"
- Make distractors believable but definitively incorrect based on the actual findings
- Ensure distractors are diseases/findings that would have been explicitly documented if present

OUTPUT FORMAT:
Return a JSON object with exactly this structure:
{{
  "questions": [
    {{
      "question": "What is the rhythm observed on this ECG?",
      "options": [
        "Atrial fibrillation",
        "Normal sinus rhythm",
        "Atrial flutter",
        "Junctional rhythm"
      ],
      "correct_answer": "Atrial fibrillation",
      "explanation": "The ECG shows irregularly irregular rhythm with absence of distinct P waves and fibrillatory waves in the baseline, characteristic of atrial fibrillation."
    }}
  ]
}}

IMPORTANT:
- Generate as many questions as there are distinct diagnostic findings in the report
- Each question should target ONE specific diagnosis
- Do not create questions where multiple options could be correct
- Ensure all distractors are significant findings that would be mentioned if present"""

async def generate_mcq_for_report(report: str, report_id: int, modality: str) -> Dict[str, Any]:
    """
    Generate multiple-choice diagnostic questions for a single imaging report
    
    Args:
        report: The cardiology imaging report text
        report_id: Index/ID of the report in the original list
        modality: One of 'ecg', 'echo', or 'cmr'
        
    Returns:
        Dictionary with report_id and generated questions
    """
    system_prompt = create_mcq_system_prompt(modality)
    
    user_prompt = f"""Based on the following cardiology imaging report, generate multiple-choice diagnostic questions.

REPORT:
{report}

Remember:
- Generate AS MANY questions as there are diagnostic findings
- Each question must have EXACTLY ONE correct answer
- Distractors must be findings that would be explicitly mentioned if present
- Questions must be answerable from images alone
- Return valid JSON with the exact structure specified"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        
        response_text = (resp.choices[0].message.content or "").strip()
        
        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1]
            response_text = response_text.split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            response_text = response_text.split("```")[0].strip()
        
        qa_data = json.loads(response_text)
        
        # Validate structure
        if "questions" not in qa_data:
            raise ValueError("Response missing 'questions' key")
        
        # Add report_id to the result
        result = {
            "report_id": report_id,
            "modality": modality,
            "report_text": report[:200] + "..." if len(report) > 200 else report,
            "questions": qa_data["questions"],
            "status": "success"
        }
        
        print(f"✓ Successfully generated {len(qa_data['questions'])} MCQ for report {report_id}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error for report {report_id}: {str(e)}")
        print(f"Response text: {response_text}")
        return {
            "report_id": report_id,
            "modality": modality,
            "questions": [],
            "status": "failed",
            "error": f"JSON parsing error: {str(e)}"
        }
    except Exception as e:
        print(f"✗ Error processing report {report_id}: {str(e)}")
        return {
            "report_id": report_id,
            "modality": modality,
            "questions": [],
            "status": "failed",
            "error": str(e)
        }

def create_none_variant_questions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each MCQ, create a duplicate where the correct answer is replaced with "None of the other options"
    
    Args:
        results: List of results from generate_mcq_for_report
        
    Returns:
        Modified results with duplicate questions added
    """
    augmented_results = []
    
    for result in results:
        if result["status"] != "success":
            augmented_results.append(result)
            continue
        
        augmented_result = copy.deepcopy(result)
        original_questions = augmented_result["questions"]
        augmented_questions = []
        
        for q in original_questions:
            # Validate required fields
            if "question" not in q or "options" not in q or "correct_answer" not in q:
                print(f"⚠ Skipping malformed question in report {result['report_id']}: missing required fields")
                augmented_questions.append(q)
                continue
            
            # Add original question
            augmented_questions.append(q)
            
            # Create "None" variant
            none_variant = copy.deepcopy(q)
            
            # Replace correct answer with "None of the other options"
            correct_answer = none_variant["correct_answer"]
            options = none_variant["options"]
            
            # Validate correct answer is in options
            if correct_answer not in options:
                print(f"⚠ Skipping 'None' variant for report {result['report_id']}: correct answer not in options")
                continue
            
            # Remove the correct answer from options
            new_options = [opt for opt in options if opt != correct_answer]
            
            # Add "None of the other options" as the new correct answer
            new_options.append("None of the other options")
            
            none_variant["options"] = new_options
            none_variant["correct_answer"] = "None of the other options"
            
            # Handle missing explanation
            original_explanation = none_variant.get("explanation", "")
            if original_explanation:
                none_variant["explanation"] = f"The correct diagnosis is {correct_answer}, which is not listed among the other options. {original_explanation}"
            else:
                none_variant["explanation"] = f"The correct diagnosis is {correct_answer}, which is not listed among the other options."
            
            none_variant["is_none_variant"] = True
            none_variant["original_correct_answer"] = correct_answer
            
            augmented_questions.append(none_variant)
        
        augmented_result["questions"] = augmented_questions
        augmented_results.append(augmented_result)
    
    return augmented_results
    
async def generate_mcq_dataset(reports: List[str], modality: str, max_concurrent: int = 5, add_none_variants: bool = True) -> List[Dict[str, Any]]:
    """
    Generate MCQ dataset for a list of reports using concurrent processing
    
    Args:
        reports: List of cardiology imaging report strings
        modality: One of 'ecg', 'echo', or 'cmr'
        max_concurrent: Maximum number of concurrent API calls
        add_none_variants: Whether to add "None of the other options" variants
        
    Returns:
        List of dictionaries containing MCQ data for each report
    """
    # Validate modality
    if modality not in DIAGNOSTIC_CATEGORIES:
        raise ValueError(f"Invalid modality: {modality}. Must be one of: {list(DIAGNOSTIC_CATEGORIES.keys())}")
    
    print(f"\n{'#'*80}")
    print(f"GENERATING DIAGNOSTIC MCQ DATASET")
    print(f"Modality: {modality.upper()}")
    print(f"Total reports: {len(reports)}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Add 'None' variants: {add_none_variants}")
    print(f"{'#'*80}\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_generate(report: str, idx: int):
        async with semaphore:
            return await generate_mcq_for_report(report, idx, modality)
    
    # Generate tasks for all reports
    tasks = [bounded_generate(report, idx) for idx, report in enumerate(reports)]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Sort by report_id to maintain original order
    results.sort(key=lambda x: x["report_id"])
    
    # Add "None of the other options" variants
    if add_none_variants:
        print("\n" + "="*80)
        print("CREATING 'NONE OF THE OTHER OPTIONS' VARIANTS")
        print("="*80)
        results = create_none_variant_questions(results)
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    total_questions = sum(len(r["questions"]) for r in results)
    original_questions = total_questions // 2 if add_none_variants else total_questions
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(reports)}")
    print(f"Failed: {failed}/{len(reports)}")
    if add_none_variants:
        print(f"Original questions generated: {original_questions}")
        print(f"'None' variant questions: {original_questions}")
    print(f"Total questions: {total_questions}")
    
    return results

def save_mcq_dataset(results: List[Dict[str, Any]], output_file: str = "mcq_dataset.json"):
    """Save the MCQ dataset to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Dataset saved to: {output_file}")

# -----------------------------
# Example usage
# -----------------------------
async def main():
    # SET MODALITY HERE - Choose one of: 'ecg', 'echo', 'cmr'
    MODALITY = os.getenv("MODALITY", "echo_new")  # Default to echo, can be set via environment variable
    
    # Example cardiology reports (replace with your actual reports)
    # sample_reports = pd.read_csv('ecg/ecg_diagnoses.csv').ECG_INTERP.dropna().tolist()[:50000]
    # sample_reports = pd.read_csv('study_report_matches.csv').note_text.dropna().tolist()
    sample_reports = pd.read_csv('/home/masadi/echo_new/new_test_reports.csv').note_text.dropna().tolist()

    
    # Generate MCQ dataset
    results = await generate_mcq_dataset(
        sample_reports, 
        modality=MODALITY, 
        max_concurrent=200,
        add_none_variants=True  # Set to False if you don't want "None" variants
    )
    
    # Save to file with modality in filename
    output_file = f"cardiology_mcq_dataset_{MODALITY}.json"
    save_mcq_dataset(results, output_file)
    
    # Print sample questions from first report
    if results and results[0]["status"] == "success" and results[0]["questions"]:
        print(f"\n{'='*80}")
        print(f"SAMPLE MCQ (Report 0 - {MODALITY.upper()}):")
        print(f"{'='*80}")
        
        # Show first original question and its "None" variant
        for i in range(min(2, len(results[0]["questions"]))):
            q = results[0]["questions"][i]
            is_none = q.get("is_none_variant", False)
            variant_label = "[NONE VARIANT]" if is_none else "[ORIGINAL]"
            
            print(f"\n{variant_label} Question {i+1}: {q['question']}")
            for j, opt in enumerate(q['options'], 1):
                marker = "✓" if opt == q['correct_answer'] else " "
                print(f"  {marker} {j}. {opt}")
            print(f"Explanation: {q['explanation']}")

if __name__ == "__main__":
    asyncio.run(main())
