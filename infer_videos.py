from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import os
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
import pandas as pd
import re

# Set to True to mock the inference process. Useful for debugging.
MOCK_INFERENCE = False

VIDEOS_PATH = "./data/7"
VIDEO_QUESTIONS_PATH = "./q_7.csv"
OUTPUT_PATH = "./output_7.csv"

SHORT_PREFIX = "The red box shows a person’s name. Please answer by choosing one of the options."
LONG_PREFIX = """You are a multimodal understanding model. The input video contains scenes where characters are annotated with red bounding boxes labeled as 1, 2, etc. Your task is to answer questions based on the visual content of the video. Use the red bounding boxes and their labels to determine which character performs the described action or matches the described attributes.

Focus on:

- The visual content inside the red bounding boxes.
- The labels (1, 2, etc.) to identify characters.
- Attributes or actions described in the question.

Question: """

PREFIX = SHORT_PREFIX

class ModelWrapper:
    def __init__(self, model, tokenizer, image_processor, context_len):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len

def infer(message_prefix: str, 
          message: str, 
          video_path: str, 
          model_wrapper: ModelWrapper) -> str:
    """
    Infer uses the LongVU model to answer questions about a video.

    Args:
    - message_prefix: The prefix to add to the question.
    - message: The question to ask
    - video_path: The path to the video to analyze.

    Returns:
    - The model answer to the message.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
    video = []
    for frame_index in frame_indices:
        img = vr[frame_index].asnumpy()
        video.append(img)
    video = np.stack(video)
    image_sizes = [video[0].shape[:2]]
    video = process_images(video, model_wrapper.image_processor, model_wrapper.model.config)
    video = [item.unsqueeze(0) for item in video]

    qs = DEFAULT_IMAGE_TOKEN + "\n" + message_prefix + " " + message
    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, model_wrapper.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model_wrapper.model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, model_wrapper.tokenizer, input_ids)

    print("Start infering...")
    if MOCK_INFERENCE:
        return "MOCK"
    
    with torch.inference_mode():
        output_ids = model_wrapper.model.generate(
            input_ids,
            images=video,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return model_wrapper.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
@dataclass
class VideoQuestion:
    video_name: str
    question_id: str
    question: str
    # Answers are in the format of "a. answer1 b. answer2 c. answer3"
    answers: str 
    # The correct answer is the letter of the correct answer.
    correct_answer: str

    def __str__(self):
        return f"Video: {self.video_name} | Question: {self.question} | Answers: {self.answers} | Correct answer: {self.correct_answer} | Question ID: {self.question_id}"

def load_video_questions_from_csv_file(video_questions_path: str) -> List[VideoQuestion]:
    """
    Load video questions from a CSV file.

    Args:
    - video_questions_path: The path to the CSV file containing the questions.

    Returns:
    - A list of VideoQuestion objects.
    """
    data = pd.read_csv(video_questions_path)
    
    video_questions = []
    for _, row in data.iterrows():
        video_questions.append(VideoQuestion(
            video_name=row["video_name"],
            question_id=row["question_id"],
            question=row["question"],
            answers=row["answers"],
            correct_answer=row["correct_answer"]
        ))

    return video_questions

def group_video_questions_by_video(video_questions: List[VideoQuestion]) -> dict:
    """
    Group video questions by video name.

    Args:
    - video_questions: A list of VideoQuestion objects.

    Returns:
    - A dictionary where the key is the video name and the value is a list of VideoQuestion objects.
    """
    video_questions_by_video = {}
    for video_question in video_questions:
        video_questions_by_video.setdefault(video_question.video_name, [])
        video_questions_by_video[video_question.video_name].append(video_question)
    
    return video_questions_by_video

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

def main():
    # Load the model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        "./checkpoints/longvu_qwen", None, "cambrian_qwen",
    )

    model.eval()
    print("Model evaluated successfully!")

    model_wrapper = ModelWrapper(model, tokenizer, image_processor, context_len)

    # Load the video questions
    video_questions = load_video_questions_from_csv_file(VIDEO_QUESTIONS_PATH)
    video_questions_by_video = group_video_questions_by_video(video_questions)
    print(f"Questions: {len(video_questions)} \nVideos: {len(video_questions_by_video)} \nLoaded: {VIDEO_QUESTIONS_PATH}")

    print(f"Prefix set to: {PREFIX}")
    print(f"Videos path: {VIDEOS_PATH}")
    
    question_id_to_model_answer = {}
    for video_name in os.listdir(VIDEOS_PATH):
        question_id_to_model_answer_per_video = {}

        video_path = os.path.join(VIDEOS_PATH, video_name)
        print(f"Analysing video: {video_path}")

        for qs in video_questions_by_video.get(video_name, []):
            print(qs)
            message = f"{qs.question} {qs.answers}"
            pred = infer(PREFIX, message, video_path, model_wrapper)
            print(f"Model result: {pred}")
            question_id_to_model_answer[qs.question_id] = pred
            question_id_to_model_answer_per_video[qs.question_id] = pred

        # Adds the new question ids to a file after each video
        with open(OUTPUT_PATH, "a") as f:
            for question_id, model_answer in question_id_to_model_answer_per_video.items():
                f.write(f"{question_id},{model_answer}\n")

    print(f"Model answers saved to: {OUTPUT_PATH}")

    # Compare the model answers with the correct answers
    print("Comparing model answers with the correct answers")
    correct_answers = 0
    total_questions = 0
    for question_id, answer in question_id_to_model_answer.items():
        for video_question in video_questions:
            if video_question.question_id == question_id:
                # Find the first letter of the answer and lower case
                if get_first_letter(answer.lower()) == video_question.correct_answer:
                    correct_answers += 1
                total_questions += 1
                break

    print(f"Correct answers: {correct_answers} / {total_questions}")

    print("Done")

if __name__ == "__main__":
    main()