from sys import __stdout__
from typing import Dict, List, Set, Tuple

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_face_mesh = mp.solutions.face_mesh  # type: ignore
mp_face_mesh_connections = mp.solutions.face_mesh_connections  # type: ignore

from settings import *


def plot_selected_features(
    image_file: str, selected_landmark_idx: List[int], output_file_name: str = ""
) -> None:
    vertices_sets: Dict[str, Set[Tuple[int, int]]] = {
        "FACEMESH_FACE_OVAL": mp_face_mesh_connections.FACEMESH_FACE_OVAL,
        "FACEMESH_LIPS": mp_face_mesh_connections.FACEMESH_LIPS,
        "FACEMESH_LEFT_EYE": mp_face_mesh_connections.FACEMESH_LEFT_EYE,
        "FACEMESH_LEFT_IRIS": mp_face_mesh_connections.FACEMESH_LEFT_IRIS,
        "FACEMESH_LEFT_EYEBROW": mp_face_mesh_connections.FACEMESH_LEFT_EYEBROW,
        "FACEMESH_RIGHT_EYE": mp_face_mesh_connections.FACEMESH_RIGHT_EYE,
        "FACEMESH_RIGHT_EYEBROW": mp_face_mesh_connections.FACEMESH_RIGHT_EYEBROW,
        "FACEMESH_RIGHT_IRIS": mp_face_mesh_connections.FACEMESH_RIGHT_IRIS,
    }

    uniform_circle_radius: int = 1

    drawing_spec_oval = mp_drawing.DrawingSpec(
        color=[255, 0, 0], thickness=1, circle_radius=uniform_circle_radius
    )
    drawing_spec_lips = mp_drawing.DrawingSpec(
        color=[0, 255, 0], thickness=1, circle_radius=uniform_circle_radius
    )
    drawing_spec_eyes = mp_drawing.DrawingSpec(
        color=[0, 0, 255], thickness=1, circle_radius=uniform_circle_radius
    )
    drawing_spec_iris = mp_drawing.DrawingSpec(
        color=[255, 255, 0], thickness=1, circle_radius=uniform_circle_radius
    )
    drawing_spec_eyebrow = mp_drawing.DrawingSpec(
        color=[0, 255, 255], thickness=1, circle_radius=uniform_circle_radius
    )
    drawing_spec_norm = mp_drawing.DrawingSpec(
        color=[255, 0, 255], thickness=1, circle_radius=uniform_circle_radius
    )
    all_points = mp_drawing.DrawingSpec(
        color=[255, 255, 255], thickness=1, circle_radius=uniform_circle_radius
    )
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for idx, file in enumerate([image_file]):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            face_landmarks = results.multi_face_landmarks[0]
            # oval
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_oval,
            )

            # lips
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_lips,
            )

            # eyes
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_eyes,
            )

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_eyes,
            )

            # iris
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_iris,
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_iris,
            )

            # eyebrow
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_LEFT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_eyebrow,
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh_connections.FACEMESH_RIGHT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_eyebrow,
            )

            # All points
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=None,
                landmark_drawing_spec=all_points,
                connection_drawing_spec=None,
            )

            s = selected_landmark_idx
            c = [1 for i in range(len(s))]

            # selected
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=list(zip(c, s)),
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_norm,
            )

            filename = "./annotated_image-" + str(idx) + ".png"
            if output_file_name != "":
                filename = f"./{output_file_name}.png"
            cv2.imwrite(filename, annotated_image)
            print("save to", filename)
