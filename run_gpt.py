from openai import OpenAI
from openai import AzureOpenAI
import os
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Evals'))
from data.utils import video_to_base64_simple, video_to_base64_frames_ungridded, image_to_base64
import tqdm
import random
import argparse

# Assumed data format:
# {
#   "id": "E3101232-E3101232",
#   "modality": "CMR or Echo or ECG",
#   "type": "Multiple-choice or Open-ended",
#   "question": "What is the left ventricular size and function?",
#   "images": ["https://example.com/image.jpg"],
#   "videos": ["https://example.com/video.mp4"],
#   "gts": "The left ventricular size is 50mm and the function is 50%.",
#   "gts_reasoning": "The left ventricular size is 50mm and the function is 50%." or "",
#   "gts_answer": "A. normal" or "The left ventricular size is 50mm and the function is 50%.",
#   "MMC": "<think>reasoning</think> A. answer",
#   "MMC_reasoning": "The left ventricular size is 50mm and the function is 50%." or "",
#   "MMC_answer": "A. normal" or "The left ventricular size is 50mm and the function is 50%.",
#   "metadata": {
#     "category": "functionality",
#     "score": 1.0,
#     "importance": "high"
#   }
# }


ECHO_SYSTEM_PROMPT = """You are a board-level cardiologist expert in echocardiography.

Context:
- Input images are arranged in a tiled grid per frame; each tile shows one echocardiographic view or sequence (e.g., PLAX, PSAX, A4C, A2C, A3C, subcostal, apical long-axis, or Doppler/Color Doppler views). Each tile's label is printed on the image.
- The provided frames are uniformly sampled across the cardiac cycle. Each frame index corresponds across all tiles (frame n = nth frame for every view).

Instructions:
1) Read the tile labels carefully. Interpret anatomical structures, wall motion, and valve motion patterns across frames.
2) Base your interpretation only on visible evidence. Choose the most appropriate answer from the options provided.
3) For 2D cine sequences, assess differences between end-diastole and end-systole (e.g., chamber size, wall thickening, wall motion abnormalities, valve opening/closure, septal motion).
4) For Doppler or Color Doppler images, describe flow patterns, gradients, and regurgitant or stenotic jets, and localize abnormalities to specific valves or chambers.
5) Integrate findings across multiple views when applicable (e.g., confirm regional wall motion abnormalities, estimate systolic function, or assess pericardial effusion).

Output format:

If the question is multiple-choice:
Respond strictly in the format of the example below:
(Example: <think>There is apical hypokinesis seen on A4C and A2C views.</think> C. Regional wall motion abnormality)

If the question is free-text:
Respond strictly as:
<think>brief clinical reasoning or justification</think> direct short answer
(Example: <think>Normal LV size and systolic thickening in PLAX, PSAX, and apical views.</think> Normal left ventricular systolic function)
"""

CMR_SYSTEM_PROMPT_UNGRIDDED = """You are a board-level cardiologist expert in cardiac MRI (CMR).

Context:
- Input images are frames from different CMR views or sequences (e.g., 2CH, 3CH, 4CH cine, SAX stack cine, LGE, T2, mapping). 
- There are 16 frames per view or sequence that are sampled uniformly across the cardiac cycle for cine sequences and the same frame repeated 16 times for static images.
- The CMR views or sequences are labelled on the images and come one after the other.

Instructions:
1) Read the CMR views or sequences labels carefully. Interpret anatomical structures and cardiac motion patterns across frames.
2) Base your interpretation only on visible evidence. Choose the most appropriate answer from the options provided.
3) For cine sequences, infer differences between end-diastole and end-systole (e.g., chamber size, wall thickening, septal motion, valve excursion).
4) For LGE, T2, or mapping images, describe signal abnormalities and localize to vascular territory or myocardial wall segments.

Output format:
- If the question is multiple-choice:
  Respond strictly as the format of the example below:
  (Example: <think>There is transmural LGE in the anterior wall consistent with LAD infarction.</think> C. Myocardial infarction)

- If the question is free-text:
  Respond strictly as:
  <think>brief clinical reasoning or justification</think> direct short answer
  (Example: <think>Normal LV size and systolic function on SAX and 4CH cine.</think> Normal left ventricular function)
"""


CMR_SYSTEM_PROMPT = """You are a board-level cardiologist expert in cardiac MRI (CMR).

Context:
- Input images are arranged in a tiled grid per frame; each tile shows one CMR view or sequence (e.g., 2CH, 3CH, 4CH cine, SAX stack cine, LGE, T2, mapping). Each tile's label is printed on the image.
- The provided frames are uniformly sampled across the cardiac cycle. Each frame index corresponds across all tiles (frame n = nth frame for every view).

Instructions:
1) Read the tile labels carefully. Interpret anatomical structures and cardiac motion patterns across frames.
2) Base your interpretation only on visible evidence. Choose the most appropriate answer from the options provided.
3) For cine sequences, infer differences between end-diastole and end-systole (e.g., chamber size, wall thickening, septal motion, valve excursion).
4) For LGE, T2, or mapping images, describe signal abnormalities and localize to vascular territory or myocardial wall segments.

Output format:
- If the question is multiple-choice:
  Respond strictly as the format of the example below:
  (Example: <think>There is transmural LGE in the anterior wall consistent with LAD infarction.</think> C. Myocardial infarction)

- If the question is free-text:
  Respond strictly as:
  <think>brief clinical reasoning or justification</think> direct short answer
  (Example: <think>Normal LV size and systolic function on SAX and 4CH cine.</think> Normal left ventricular function)
"""

ECG_SYSTEM_PROMPT  = """
You are a board-level cardiologist expert in electrocardiography (ECG).

Context:
- Input images are ECG waveforms from different leads (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6).
- The ECG waveforms are labelled on the images and come one after the other.

Instructions:
    1) Read the ECG waveforms carefully. Interpret the waveforms to answer the question.
    2) Base your interpretation only on visible evidence. Choose the most appropriate answer from the options provided.
    3) For each waveform, assess the morphology, amplitude, and duration.
    4) Integrate findings across multiple leads when applicable (e.g., assess for bundle branch block, atrioventricular conduction delay, etc.).

    Output format:
- If the question is multiple-choice:
    Respond strictly as the format of the example below:
    (Example: <think>The QRS duration exceeds 120 ms, the morphology shows a broad, notched R wave in leads I and V6 and a deep, wide S wave in leads V1–V3, fulfilling criteria for left bundle branch block.</think> C. Left bundle branch block)

- If the question is free-text:
  Respond strictly as:
  <think>brief clinical reasoning or justification</think> direct short answer
  (Example: <think>The QRS duration exceeds 120 ms, the morphology shows a broad, notched R wave in leads I and V6 and a deep, wide S wave in leads V1–V3, fulfilling criteria for left bundle branch block.</think> Left bundle branch block)
"""


MULTIMODAL_SYSTEM_PROMPT = """
You are a board-level cardiologist expert in cardiology.

Context:
- Input images are from different modalities (echo, CMR, ECG).
- The images are labelled on the images and come one after the other.

Instructions:
1) Read the images carefully. Interpret the images to answer the question.
2) Base your interpretation only on visible evidence. Choose the most appropriate answer from the options provided.
3) Integrate findings across multiple modalities when applicable (e.g., assess for regional wall motion abnormalities, estimate systolic function, or assess pericardial effusion).
- If the question is multiple-choice:
    Respond strictly as the format of the example below:
    (Example: <think>The imaging findings on echo and CMR show: Echo findings: The images show an LVEF of 66% with normal wall motion, indicating normal systolic performance. CMR findings: Quantitative measurement shows RV EF of 52.7%, which is within the normal range. Integrated assessment requires synthesizing information from both modalities. The ECG demonstrates: Sinus rhythm replaces the electronic atrial pacemaker, so no pacemaker spikes are seen on the tracing. Both findings must be correctly identified for accurate diagnosis.</think> A. Normal left ventricular systolic function with specific ECG findings)

- If the question is free-text:
  Respond strictly as:
  <think>brief clinical reasoning or justification</think> direct short answer
  (Example: <think>The imaging findings on echo and CMR show: Echo findings: The images show an LVEF of 66% with normal wall motion, indicating normal systolic performance. CMR findings: Quantitative measurement shows RV EF of 52.7%, which is within the normal range. Integrated assessment requires synthesizing information from both modalities. The ECG demonstrates: Sinus rhythm replaces the electronic atrial pacemaker, so no pacemaker spikes are seen on the tracing. Both findings must be correctly identified for accurate diagnosis.</think> Normal left ventricular systolic function with specific ECG findings)
"""


SYSTEM_PROMPT = {
    "ECHO_IMAGE": ECHO_SYSTEM_PROMPT,
    "CMR_IMAGE": CMR_SYSTEM_PROMPT,
    "ECG_IMAGE": ECG_SYSTEM_PROMPT,
    "MULTIMODAL": MULTIMODAL_SYSTEM_PROMPT,
}


model = "gpt-5"


securegpt_api_key = os.environ.get("SECUREGPT_API_KEY")
headers = {
    'Ocp-Apim-Subscription-Key': securegpt_api_key,
    'Content-Type': 'application/json',
}
API_VERSION = '2024-12-01-preview'
DEPLOYMENT = model # gpt-5-mini # gpt-5-nano
OPENAI_ENDPOINT= os.environ.get('OPENAI_ENDPOINT')  # e.g. "openai123"
URL = f'https://apim.stanfordhealthcare.org/openai-eastus2/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}'

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=URL,
    azure_deployment=DEPLOYMENT,
    default_headers=headers,
    azure_ad_token=securegpt_api_key,
)




# def run_gpt(model, item, base64Frames):
#     print(len(base64Frames))
#     response = client.responses.create( # TODO: add a system prompt here 
#         model=model,
#         input=[
#             {
#                 "role": "system",
#                 "content": [
#                     {
#                         "type": "input_text",
#                         "text": SYSTEM_PROMPT[MODE]
#                     }
#                 ]
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "input_text",
#                         "text": (
#                             f"Answer the following question: \n{item['question']} \n"
#                         )
#                     },
#                     *[
#                         {
#                             "type": "input_image",
#                             "image_url": f"data:image/jpeg;base64,{frame}"
#                         }
#                         for frame in base64Frames
#                     ]
#                 ]
#             }
#         ],
#     )

#     return response.output_text

def run_gpt(model, item, base64Frames, MODE):
    print(len(base64Frames))
    response = client.chat.completions.create( 
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT[MODE]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Answer the following question: \n{item['question']} \n"
                        )
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame}"
                            }
                        }
                        for frame in base64Frames
                    ]
                ]
            }
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GPT evaluation on cardiology test data"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ECHO_IMAGE", "CMR_IMAGE", "ECG_IMAGE", "MULTIMODAL"],
        help="Mode for evaluation (determines system prompt and image processing)"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to input JSON data file"
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to output JSON results file"
    )
    
    args = parser.parse_args()
    
    MODE = args.mode
    data_path = args.data_path
    results_path = args.results_path

    print(f"Running evaluation with:")
    print(f"  MODE: {MODE}")
    print(f"  Data path: {data_path}")
    print(f"  Results path: {results_path}")
    print()

    with open(data_path, 'r') as f:
        multimodal_test = json.load(f).copy()

    # MMC_total_correct = 0
    model_total_correct = 0
    total_questions = 0
    for item in tqdm.tqdm(multimodal_test):
        total_questions += 1
        # cmr
        # base64Frames = video_to_base64_frames_ungridded(item["videos"][0])
        # print(item["videos"][0])
        # base64Frames = [tile[::15] for tile in base64Frames]
        # base64Frames = [frame for tile in base64Frames for frame in tile]
        # base64Frames = video_to_base64_simple(item["videos"][0])[::15]

        # echo
        # base64Frames = video_to_base64_simple(item["videos"][0])

        # ecg
        # base64Frames = [image_to_base64(item["images"][0])]

        # multimodal
        # base64Frames = []
        # if "images" in item and len(item["images"]) > 0:
        #     base64Frames += [image_to_base64(item["images"][0])]
        # if "videos" in item and len(item["videos"]) > 0:
        #     base64Frames += video_to_base64_simple(item["videos"][0]) # echo
        # if "videos" in item and len(item["videos"]) > 1: 
        #     base64Frames += video_to_base64_simple(item["videos"][1])[::15] # CMR

        if MODE == "ECG_IMAGE":
            base64Frames = [image_to_base64(item["images"][0])]
        elif MODE == "CMR_IMAGE":
            base64Frames = video_to_base64_simple(item["videos"][0])[::15]
        elif MODE == "ECHO_IMAGE":
            base64Frames = video_to_base64_simple(item["videos"][0])
        elif MODE == "MULTIMODAL":
            base64Frames = []
            if "images" in item and len(item["images"]) > 0:
                base64Frames += [image_to_base64(item["images"][0])]
            if "videos" in item and len(item["videos"]) > 0:
                base64Frames += video_to_base64_simple(item["videos"][0]) # echo
            if "videos" in item and len(item["videos"]) > 1: 
                base64Frames += video_to_base64_simple(item["videos"][1])[::15] # CMR


        print(len(base64Frames))
        try:
            item[f"{model}"] = run_gpt(model, item, base64Frames, MODE)
        except Exception as e:
            print(e)
            item[f"{model}"] = "API ERROR"



        if item['type'] == 'Multiple-choice':
            try:
                item[f"{model}_reasoning"] = item[f"{model}"].split('</think>')[0].strip()
            except:
                item[f"{model}_reasoning"] = "format error"
            try:
                item[f"{model}_answer"] = item[f"{model}"].split('</think>')[1].strip()
            except:
                item[f"{model}_answer"] = "format error"

            if item[f"{model}_answer"].replace(' ', '').replace('\n', '') == item["gts_answer"].replace(' ', '').replace('\n', ''):
                model_total_correct += 1
            # if item["MMC_answer"].replace(' ', '').replace('\n', '') == item["gts_answer"].replace(' ', '').replace('\n', ''):
            #     MMC_total_correct += 1

            print("--------------------------------")
            print("--------------------------------")
            print(item["question"])
            print("Ground truth: ", item["gts_answer"])
            # print("MMC answer: ", item["MMC_answer"])
            print(f"{model} answer: ", item[f"{model}_answer"])
            print("--------------------------------")
            # print(f"MMC total correct: {MMC_total_correct} from {total_questions} questions")
            print(f"Model total correct: {model_total_correct} from {total_questions} questions")
            print("--------------------------------")
            print("--------------------------------")
        elif item['type'] == 'Open-ended':
            print("--------------------------------")
            print("--------------------------------")
            print(item["question"])
            print("Ground truth: ", item["gts"])
            print(f"{model} answer: ", item[f"{model}"])
            print("--------------------------------")
            print("--------------------------------")




    with open(results_path, 'w') as f:
        json.dump(multimodal_test, f, indent=4)


