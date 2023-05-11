import csv
from pathlib import Path
import cv2

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_face_mesh = mp.solutions.face_mesh  # type: ignore
mp_face_mesh_connections = mp.solutions.face_mesh_connections  # type: ignore

from settings import *

## %%script echo skip

BLENDSHAPE_I = []
WEIGHTS = []
IMAGE_FILES = []

csv_file = open(BLENDSHAPE_FILE, "r")
reader = csv.reader(csv_file, skipinitialspace=True, delimiter=",")
## skip header
next(reader, None)
for line in reader:
    if not Path(line[2]).exists():
        print(line[2])
        print("File not found")
        continue
    BLENDSHAPE_I.append(int(line[0]))
    WEIGHTS.append(int(line[1]))
    IMAGE_FILES.append(line[2])
csv_file.close()

train_file = open(TRAIN_FILE, "w")
writer = csv.writer(train_file, delimiter=",", quoting=csv.QUOTE_ALL)
writer.writerow(HEADERS)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        # The 1-D array of each y_i of Y.
        weights = [0 for i in range(0, N_BLENDSHAPE)]
        weights[BLENDSHAPE_I[idx] - FIRST_BLENDSHAPE_IDX] = WEIGHTS[idx]
        arr = [BLENDSHAPE_I[idx], weights]

        # Assume to have only one face in tracking,
        # Following is equal to results.multi_face_landmarks[0]
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                arr.append([landmark.x, landmark.y, landmark.z])
            writer.writerow(arr)

train_file.close()
