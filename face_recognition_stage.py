from dataclasses import dataclass
from typing import List
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
import os

VIDEOS_PATH = "./videos"
OUTPUT_PATH = "./output"
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

@dataclass
class AnnotationConfig:
    known_faces: List[any]

    # The annotations are listed in the same order as the known_faces
    # and represents the names which will be displayed on the video
    annotations: np.array

    input_filename: str
    output_filename: str

    def __init__(self, 
                 input_filename, 
                 output_filename, 
                 known_faces, 
                 annotations):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.known_faces = known_faces
        self.annotations = annotations

    def __str__(self):
        return f"AnnotationConfig(input_filename={self.input_filename}, output_filename={self.output_filename})"

def annotate_video(input_movie, annotation_config, output_movie):
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(annotation_config.known_faces, face_encoding, tolerance=0.50)
            names = annotation_config.annotations[match]
            face_names.extend(str(name) for name in names)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Write the resulting image to the output video file
        # print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()

def create_annotation_config():
    annotation_configs = []

    for video_dir in os.listdir(VIDEOS_PATH):
        files_in_dir = os.listdir(os.path.join(VIDEOS_PATH, video_dir))

        video_file = list(filter(lambda x: x.endswith(VIDEO_SUFFIXES), files_in_dir))
        image_files = list(filter(lambda x: x.endswith(IMAGE_SUFFIXES), files_in_dir))

        if len(video_file) != 1:
            print(f"Expected 1 video file in {video_dir}, but found {len(video_file)}")
            continue

        if len(image_files) == 0:
            print(f"Expected at least 1 image file in {video_dir}, but found {len(image_files)}")
            continue

        input_filename = os.path.join(VIDEOS_PATH, video_dir, video_file[0])
        output_filename = os.path.join(OUTPUT_PATH, video_file[0])

        known_faces = []
        annotations = np.empty(len(image_files), dtype=object)

        for i, image_file in enumerate(image_files):
            image = face_recognition.load_image_file(os.path.join(VIDEOS_PATH, video_dir, image_file))
            face_encoding = face_recognition.face_encodings(image)[0]

            known_faces.append(face_encoding)
            annotations[i] = image_file.split(".")[0]

        annotation_configs.append(AnnotationConfig(known_faces=known_faces,
                                                   annotations=annotations,
                                                   input_filename=input_filename,
                                                   output_filename=output_filename))
        
    return annotation_configs

def main():
    annoation_configs = create_annotation_config()
    print(f"Found {len(annoation_configs)} videos to annotate")

    for config in tqdm(annoation_configs):
        print(config)

        # Open the input movie file
        input_movie = cv2.VideoCapture(config.input_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # For now, all videos assume to be 25 fps and (640,360) wide.
        # You can use ffmpeg to convert your videos before using the script
        # using the follwing command:
        #       ffmpeg -i input_file -vcodec h264 -acodec aac -r 25 -vf scale=640:360 output_file
        output_movie = cv2.VideoWriter(config.output_filename, fourcc, 25, (640, 360))

        annotate_video(input_movie=input_movie, 
                       annotation_config=config, 
                       output_movie=output_movie)

    print("Done")

if __name__ == "__main__":
    main()