# pip install openai>=1.30.0
import os
import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI

import pandas as pd

# -----------------------------
# vLLM / OpenAI-compatible client
# -----------------------------
API_BASE = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("LV_FACTS_MODEL", "openai/gpt-oss-120b")
client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)

# -----------------------------
# Modality-specific categories
# -----------------------------
CATEGORIES = {
    "ecg": {
        "rhythm": "Rate, regularity, sinus vs non-sinus, ectopy",
        "conduction": "PR, QRS, QT/QTc; AV block; BBB; fascicular block; pre-excitation",
        "axis": "P/QRS/T axis; LAD/RAD/indeterminate",
        "wave_morphology": "P/QRS/ST/T/U features including R-wave progression (transition, poor progression, early transition)",
        "ischemia_infarction": "STEMI/NSTEMI patterns, acute vs old MI, reciprocal change",
        "voltage_hypertrophy_strain": "LVH, RVH, atrial enlargement; low voltage",
        "arrhythmia": "AF/AFL, SVT, VT, pauses, bigeminy/trigeminy, junctional, AIVR, MAT, etc.",
        "devices_pacing": "Atrial/ventricular/dual-chamber paced rhythm, pacemaker malfunction",
        "technical_quality_artifact": "Lead reversal, artifact, noise",
        "pattern_phenotype": "Early repolarization, Brugada pattern, pericarditis, digitalis effect (recognizable morphologic phenotypes)"
    },
    "echo": {
        "anatomy": "Chambers, septa, aorta, pericardium",
        "function": "Systolic, diastolic, wall motion, strain, stress/Takotsubo",
        "pathology": "Hypertrophy, cardiomyopathy, thrombus, vegetations, tumors, pericardial effusion, dissection, tamponade",
        "valves": "Morphology, stenosis, regurgitation, mechanical/bioprosthetic valves, TAVR/TMVR/TEER",
        "flow": "Gradients, shunts, regurgitant jets, pulmonary/hepatic venous flow",
        "measurements": "Dimensions, valve areas, gradients, pressures, EF",
        "prosthetic_valves_devices": "Pacemakers, clips",
        "quality_technical": "Study quality, limitations, acoustic shadowing, use of contrast agents"
    },
    "cmr": {
        "anatomy": "Ventricles, atria, myocardium, pericardium, great vessels, LV mass",
        "function": "Volumes, ejection fraction, wall motion",
        "pathology": "Hypertrophy, cardiomyopathies, aneurysms, thrombus, masses",
        "tissue_characterization": "Late gadolinium enhancement, edema, fat infiltration, fibrosis",
        "flow": "Shunts, regurgitation, aortic/pulmonary flow (phase contrast)",
        "measurements": "Chamber volumes, wall thickness, stroke volume, flow quantification, LV mass",
        "perfusion": "First-pass perfusion abnormalities, ischemia/infarction",
        "valves": "Morphology, stenosis, regurgitation, mechanical/bioprosthetic valves, TAVR/TMVR/TEER"
    }
}

CATEGORIES["echo_new"] = CATEGORIES["echo"]

# -----------------------------
# System prompt generation
# -----------------------------
def create_system_prompt(modality: str) -> str:
    """
    Generate modality-specific system prompt
    
    Args:
        modality: One of 'ecg', 'echo', or 'cmr'
        
    Returns:
        System prompt string
    """
    if modality not in CATEGORIES:
        raise ValueError(f"Invalid modality: {modality}. Must be one of: {list(CATEGORIES.keys())}")
    
    categories = CATEGORIES[modality]
    categories_text = "\n".join([f"- {key}: {value}" for key, value in categories.items()])
    
    return f"""You are an expert in cardiology imaging ({modality.upper()}). Your task is to generate visual question-answer pairs based on imaging reports.

CRITICAL REQUIREMENTS:
1. Questions MUST be answerable by looking at the images alone - no questions requiring patient history, prior studies, or clinical context
2. Write questions and answers as if someone is directly observing and describing the images
3. NEVER use phrases like: "mentioned", "not mentioned", "reported", "not reported", "according to", "the report shows", "described in"
4. Use present tense and direct observation language: "The left ventricle is...", "I see...", "There is...", "The ejection fraction appears..."
5. If the study report does not match the imaging modality ({modality.upper()}), return an empty list of questions.

QUESTION CATEGORIES (assign one per question):
{categories_text}

IMPORTANCE LEVELS (assign one per question):
- critical: Life-threatening findings or major diagnostic information
- high: Important clinical findings that affect management
- moderate: Relevant clinical details
- low: Minor or descriptive findings

OUTPUT FORMAT:
Return a JSON object with exactly this structure:
{{
  "questions": [
    {{
      "question": "What is the left ventricular ejection fraction?",
      "answer": "The left ventricular ejection fraction is approximately 45-50%, indicating mildly reduced systolic function.",
      "category": "function",
      "importance": "high"
    }}
  ]
}}

Generate as many questions as possible covering different categories and importance levels. 
Focus more on pathology questions when abnormalities are present.
If you cannot generate questions, generate as many high-quality questions as possible.
Stick to the information present and do not hallucinate. Specifically do not hallucinate about the presence of pathology when it is not present. and do not hallucinate about numbers when they are not present.
Do not hallucinate about lead information when it is not present directly in the report."""

async def generate_qa_for_report(report: str, report_id: int, modality: str) -> Dict[str, Any]:
    """
    Generate 15 VQA pairs for a single imaging report
    
    Args:
        report: The cardiology imaging report text
        report_id: Index/ID of the report in the original list
        modality: One of 'ecg', 'echo', or 'cmr'
        
    Returns:
        Dictionary with report_id and generated questions
    """
    system_prompt = create_system_prompt(modality)
    
    user_prompt = f"""Based on the following cardiology imaging report, generate as many visual questions and answers as possible.

REPORT:
{report}

Remember:
- Questions must be answerable from images alone
- Use direct observational language
- No references to "the report" or "mentioned"
- Diverse categories and importance levels
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
            max_tokens=2500,
        )
        
        response_text = (resp.choices[0].message.content or "").strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
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
            "report_text": report[:200] + "..." if len(report) > 200 else report,  # Store truncated report for reference
            "questions": qa_data["questions"],
            "status": "success"
        }
        
        print(f"✓ Successfully generated {len(qa_data['questions'])} QA pairs for report {report_id}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error for report {report_id}: {str(e)}")
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

async def generate_vqa_dataset(reports: List[str], modality: str, max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Generate VQA pairs for a list of reports using concurrent processing
    
    Args:
        reports: List of cardiology imaging report strings
        modality: One of 'ecg', 'echo', or 'cmr'
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of dictionaries containing VQA data for each report
    """
    # Validate modality
    if modality not in CATEGORIES:
        raise ValueError(f"Invalid modality: {modality}. Must be one of: {list(CATEGORIES.keys())}")
    
    print(f"\n{'#'*80}")
    print(f"GENERATING VQA DATASET")
    print(f"Modality: {modality.upper()}")
    print(f"Total reports: {len(reports)}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"{'#'*80}\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_generate(report: str, idx: int):
        async with semaphore:
            return await generate_qa_for_report(report, idx, modality)
    
    # Generate tasks for all reports
    tasks = [bounded_generate(report, idx) for idx, report in enumerate(reports)]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Sort by report_id to maintain original order
    results.sort(key=lambda x: x["report_id"])
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    total_questions = sum(len(r["questions"]) for r in results)
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(reports)}")
    print(f"Failed: {failed}/{len(reports)}")
    print(f"Total questions generated: {total_questions}")
    
    return results

def save_vqa_dataset(results: List[Dict[str, Any]], output_file: str = "vqa_dataset.json"):
    """Save the VQA dataset to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Dataset saved to: {output_file}")

# -----------------------------
# Example usage
# -----------------------------
async def main():
    # SET MODALITY HERE - Choose one of: 'ecg', 'echo', 'cmr'
    MODALITY = os.getenv("MODALITY", "ecg")  # Default to echo, can be set via environment variable
    
    # Example cardiology reports (replace with your actual reports)
    sample_reports = pd.read_csv('ecg/ecg_diagnoses.csv').ECG_INTERP.dropna().tolist()[:50000]
    # sample_reports = pd.read_csv('/home/masadi/echo_new/new_test_reports.csv').note_text.dropna().tolist()
    
    # Generate VQA dataset
    results = await generate_vqa_dataset(sample_reports, modality=MODALITY, max_concurrent=200)
    
    # Save to file with modality in filename
    output_file = f"cardiology_vqa_dataset_{MODALITY}.json"
    save_vqa_dataset(results, output_file)
    
    # Print sample questions from first report
    if results and results[0]["status"] == "success" and results[0]["questions"]:
        print(f"\n{'='*80}")
        print(f"SAMPLE QUESTIONS (Report 0 - {MODALITY.upper()}):")
        print(f"{'='*80}")
        for i, qa in enumerate(results[0]["questions"][:3], 1):
            print(f"\nQ{i} [{qa['category']}] [{qa['importance']}]: {qa['question']}")
            print(f"A{i}: {qa['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
