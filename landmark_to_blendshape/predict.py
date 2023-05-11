import json
import socket
import threading
from pickle import load
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn as sk
from pandas import DataFrame

from settings import *

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_face_mesh = mp.solutions.face_mesh  # type: ignore
mp_face_mesh_connections = mp.solutions.face_mesh_connections  # type: ignore

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from configparser import ConfigParser
from pathlib import Path
from transformers import FullDistance


ADDRESS = ("localhost", 8080)
TARGET_BLENDSHAPE_INDEX = list()
FILE_DIR = Path(__file__).parent

config_reader = ConfigParser()
config_reader.read(FILE_DIR / "blendshapes.ini")
blendshape_configs = config_reader["blendshapes"]

# Those blendshapes are in used.
# Store the index of those blendshapes.
for idx, key in enumerate(blendshape_configs):
    if blendshape_configs.getboolean(key):
        TARGET_BLENDSHAPE_INDEX.append(idx)


def websocket_client():
    server_address = ADDRESS
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(server_address)
    while True:
        global blendshape_weights
        global terminate_event
        if terminate_event.is_set():
            break
        if blendshape_weights.empty():
            continue
        json_weights = json.dumps(blendshape_weights.get())
        client.sendall(json_weights.encode("utf-8"))
    client.close()


def load_models() -> List[Any]:
    """Load the saved feature selection and prediction models.

    Returns:
        List[Any]: List of estimators.
    """
    estimators = list()
    for _, blendshape_idx in enumerate(TARGET_BLENDSHAPE_INDEX):
        with open(
            f"{FILE_DIR.absolute()}/estimators/estimator-index{blendshape_idx}.pkl",
            "rb",
        ) as file:
            estimators.append(load(file))
    return estimators


def main_loop():
    estimators = load_models()
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            arr = []
            for face_landmarks in results.multi_face_landmarks:
                for a in face_landmarks.landmark:
                    arr.append(np.array([a.x, a.y, a.z]))
            predict_df = pd.DataFrame([arr], columns=HEADERS[2:])
            res = [0 for i in range(N_BLENDSHAPE)]
            for idx, estimator in enumerate(estimators):
                res[TARGET_BLENDSHAPE_INDEX[idx]] = int(
                    estimator.predict(predict_df)[0]
                )
            print(res)
            global blendshape_weights
            blendshape_weights.put(res)


if __name__ == "__main__":
    blendshape_weights = Queue()

    # Terminate on KeyboardInterrupt
    terminate_event = threading.Event()

    # Start the websocket thread
    socket_client_thread = threading.Thread(target=websocket_client)
    socket_client_thread.start()
    print("Websocket thread started.")

    try:
        main_loop()
    except KeyboardInterrupt:
        terminate_event.set()
        socket_client_thread.join()
        print("Program terminated.")
        exit(0)
