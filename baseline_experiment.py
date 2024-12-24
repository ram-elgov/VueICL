import time
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from decord import cpu, VideoReader
import pandas as pd
import re
from qwen_vl_utils import process_vision_info  # Ensure this utility is available

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set to True to mock the inference process. Useful for debugging.
MOCK_INFERENCE = False

# Paths
HOME_DIR = "/home/yandex/MML2024b/ramelgov/"
VIDEOS_PATH = os.path.join(HOME_DIR, "raw_videos_characters")
VIDEO_QUESTIONS_PATH = os.path.join(HOME_DIR, "LongVU", "q_7.csv")
OUTPUT_PATH = os.path.join(HOME_DIR, "output_7.csv")

# Prefix for the prompts
SHORT_PREFIX = ""
LONG_PREFIX = """
Please answer the following multiple-choice questions on Video 1 by outputting only the correct letter and nothing more.
The numbers in the choices (1, 2, 3, 4, etc.) are shortcuts for Picture 1, Picture 2, Picture 3, etc.
"""
PREFIX = LONG_PREFIX  # Using SHORT_PREFIX as per your configuration

# Regex to extract the first English letter
ENGLISH_LETTERS = re.compile(r"[a-zA-Z]")

def get_first_letter(text: str) -> str:
    """
    Get the first letter of a text.

    Args:
    - text: The text to get the first letter from.

    Returns:
    - The first letter of the text.
    """
    match = ENGLISH_LETTERS.search(text) 
    return match.group(0) if match else ""

@dataclass
class VideoQuestion:
    video_name: str
    question_id: str
    question: str
    answers: str 
    correct_answer: str

    def __str__(self):
        return f"Video: {self.video_name} | Question ID: {self.question_id} | Question: {self.question} | Answers: {self.answers} | Correct Answer: {self.correct_answer}"

def load_video_questions_from_csv_file(video_questions_path: str) -> List[VideoQuestion]:
    """
    Load video questions from a CSV file.

    Args:
    - video_questions_path: The path to the CSV file containing the questions.

    Returns:
    - A list of VideoQuestion objects.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading questions from {video_questions_path}...")
    data = pd.read_csv(video_questions_path)
    
    video_questions = []
    for _, row in data.iterrows():
        video_name = row["video_name"]
        if video_name.endswith('.mp4'):
            video_name = video_name[:-4]  # Remove '.mp4' extension
        video_questions.append(VideoQuestion(
            video_name=video_name,
            question_id=row["question_id"],
            question=row["question"],
            answers=row["answers"],
            correct_answer=row["correct_answer"]
        ))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(video_questions)} questions.")
    return video_questions

def group_video_questions_by_video(video_questions: List[VideoQuestion]) -> dict:
    """
    Group video questions by video name.

    Args:
    - video_questions: A list of VideoQuestion objects.

    Returns:
    - A dictionary where the key is the video name and the value is a list of VideoQuestion objects.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Grouping questions by video...")
    video_questions_by_video = {}
    for video_question in video_questions:
        video_questions_by_video.setdefault(video_question.video_name, [])
        video_questions_by_video[video_question.video_name].append(video_question)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Grouped questions for {len(video_questions_by_video)} videos.")
    print(f"Video questions by video: {list(video_questions_by_video.keys())}")
    return video_questions_by_video

class ModelWrapper:
    def __init__(self, model, processor, context_len):
        self.model = model
        self.processor = processor
        self.context_len = context_len

def infer(message_prefix: str, 
          message: str, 
          image_paths: List[str],
          video_path: str, 
          model_wrapper: ModelWrapper) -> str:
    """
    Infer using the Qwen2-VL model to answer questions about a video.

    Args:
    - message_prefix: The prefix to add to the question.
    - message: The question to ask
    - image_paths: List of image file paths
    - video_path: The path to the video to analyze.

    Returns:
    - The model answer to the message.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing video: {video_path}")
    start_time = time.time()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"}
                for image_path in image_paths
            ] + [
                {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
                {"type": "text", "text": message_prefix + " " + message},
            ]
        }
    ]
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Constructed messages for inference.")
    
    # Prepare inputs for the model with add_vision_id=True
    text = model_wrapper.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = model_wrapper.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inputs prepared for the model.")
    
    # Move inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_wrapper.model.to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Moved inputs and model to {device}.")
    
    # Generate output
    try:
        if MOCK_INFERENCE:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MOCK_INFERENCE is enabled. Returning mock response.")
            return "MOCK"
        
        with torch.inference_mode():
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating output...")
            outputs = model_wrapper.model.generate(**inputs, max_new_tokens=128)
        output_text = model_wrapper.processor.decode(outputs[0], skip_special_tokens=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generated answer: {output_text.strip()}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inference time: {time.time() - start_time:.2f} seconds.")
        return output_text.strip()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CUDA out of memory. Try reducing the number of frames or resolution.")
        print(e)
        return "Inference failed due to memory error."

def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting the inference script.")
    
    # Load the Qwen2-VL model
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading Qwen2-VL model...")
    model_name = "Qwen/Qwen2-VL-7B-Instruct"  # Updated to 7B as per your example
    processor = AutoProcessor.from_pretrained(model_name)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as per your example
            attn_implementation="flash_attention_2",  # Enable FlashAttention-2
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Qwen2-VL model loaded successfully.")
    
    model_wrapper = ModelWrapper(model, processor, context_len=512)  # Adjust context_len if needed
    
    # Load the video questions
    video_questions = load_video_questions_from_csv_file(VIDEO_QUESTIONS_PATH)
    video_questions_by_video = group_video_questions_by_video(video_questions)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Prefix set to: {PREFIX}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Videos path: {VIDEOS_PATH}")
    
    # Initialize counters for accuracy
    correct_answers = 0
    total_questions = 0
    
    # Ensure output file is empty before starting
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Removed existing {OUTPUT_PATH}.")
    
    # Iterate over video folders
    for video_name in os.listdir(VIDEOS_PATH):
        video_dir = os.path.join(VIDEOS_PATH, video_name)
        video_file = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.isfile(video_file):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Video file not found: {video_file}. Skipping.")
            continue
        
        # Collect all image paths in the video directory
        image_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(".png")]
        if not image_paths:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No images found in {video_dir}. Skipping.")
            continue
        
        # Retrieve questions for this video
        questions = video_questions_by_video.get(video_name, [])
        if not questions:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No questions found for video: {video_name}. Skipping.")
            continue
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Analyzing video: {video_file} with {len(image_paths)} images and {len(questions)} questions.")
        
        # Reset the dictionary for each video to prevent mixing answers
        question_id_to_model_answer = {}
        
        for qs in questions:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing question ID: {qs.question_id}")
            message = f"{qs.question} {qs.answers}"
            pred = infer(PREFIX, message, image_paths, video_file, model_wrapper)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model result for Question ID {qs.question_id}: {pred}")
            question_id_to_model_answer[qs.question_id] = pred
        
        # Write results to the output file
        with open(OUTPUT_PATH, "a") as f:
            for question_id, model_answer in question_id_to_model_answer.items():
                f.write(f"{question_id},{model_answer}\n")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved results for video: {video_file}")
        
        # Compare the model answers with the correct answers
        for question_id, answer in question_id_to_model_answer.items():
            for video_question in video_questions:
                if video_question.question_id == question_id:
                    # Find the first letter of the answer and lower case
                    if get_first_letter(answer.lower()) == video_question.correct_answer.lower():
                        correct_answers += 1
                    total_questions += 1
                    break
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model answers saved to: {OUTPUT_PATH}")
    
    # Final accuracy report
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparing model answers with the correct answers...")
    print(f"Correct answers: {correct_answers} / {total_questions}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inference script completed.")

if __name__ == "__main__":
    main()
