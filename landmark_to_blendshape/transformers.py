from pandas import DataFrame
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FullDistance(BaseEstimator, TransformerMixin):
    def __init__(self, distance_type: str = "full"):
        """_summary_

        Args:
            is_full (bool, optional): Using the full distance or the refine distance. Defaults to True.
        """
        super().__init__()
        self.distance_type = distance_type
        self.selected_vertex_array = [
            0,
            7,
            10,
            13,
            14,
            17,
            21,
            33,
            37,
            39,
            40,
            46,
            52,
            53,
            54,
            55,
            58,
            61,
            63,
            65,
            66,
            67,
            70,
            78,
            80,
            81,
            82,
            84,
            87,
            88,
            91,
            93,
            95,
            103,
            105,
            107,
            109,
            127,
            132,
            133,
            136,
            144,
            145,
            146,
            148,
            149,
            150,
            152,
            153,
            154,
            155,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            172,
            173,
            176,
            178,
            181,
            185,
            191,
            234,
            246,
            249,
            251,
            263,
            267,
            269,
            270,
            276,
            282,
            283,
            284,
            285,
            288,
            291,
            293,
            295,
            296,
            297,
            300,
            308,
            310,
            311,
            312,
            314,
            317,
            318,
            321,
            323,
            324,
            332,
            334,
            336,
            338,
            356,
            361,
            362,
            365,
            373,
            374,
            375,
            377,
            378,
            379,
            380,
            381,
            382,
            384,
            385,
            386,
            387,
            388,
            389,
            390,
            397,
            398,
            400,
            402,
            405,
            409,
            415,
            454,
            466,
            469,
            470,
            471,
            472,
            474,
            475,
            476,
            477,
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.distance_type == "full":
            return self._full_transform(X, y)
        elif self.distance_type == "refine":
            return self._refine_transform(X, y)

    def _full_transform(self, X, y=None):
        """Produce the training set with distance between center and all the landmarks

        Args:
            input_df (DataFrame): the origin dataframe from csv file

        Returns:
            DataFrame: the training set with distance between landmarks
        """

        NOSE_IDX = 4
        TOP_DOWN_FACE = (10, 152)

        # define the column names
        new_columns = list()
        for idx, _ in enumerate(X.columns):
            new_columns.append(f"distance_{idx}")

        distance_X = DataFrame(columns=new_columns, dtype=np.float64)

        for i, row in X.iterrows():
            new_row = list()
            middle_point = np.copy(row[NOSE_IDX])
            middle_point[1] = (row[TOP_DOWN_FACE[0]][1] + row[TOP_DOWN_FACE[1]][1]) / 2
            middle_point[2] = 0
            max_distance = row[TOP_DOWN_FACE[0]] - row[TOP_DOWN_FACE[1]]
            max_distance[2] = 0
            normalised_distance = np.linalg.norm(max_distance)
            for _, landmark in enumerate(row):
                landmark_copy = np.copy(landmark)
                landmark_copy[2] = 0
                distance = np.linalg.norm(landmark_copy - middle_point)
                new_row.append(distance)
            distance_X.loc[i] = [distance / normalised_distance for distance in new_row]  # type: ignore
        return distance_X

    def _refine_transform(self, X, y=None):
        """Produce the training set with distance between center and a subset of all landmarks

        Args:
            input_df (DataFrame): the origin dataframe from csv file

        Returns:
            DataFrame: the training set with distance between landmarks
        """

        NOSE_IDX = 4
        TOP_DOWN_FACE = (10, 152)

        # define the column names
        new_columns = list()
        for idx, _ in enumerate(self.selected_vertex_array):
            new_columns.append(f"distance_{idx}")

        distance_X = DataFrame(columns=new_columns, dtype=np.float64)

        for i, row in X.iterrows():
            new_row = list()
            middle_point = np.copy(row[NOSE_IDX])
            middle_point[1] = (row[TOP_DOWN_FACE[0]][1] + row[TOP_DOWN_FACE[1]][1]) / 2
            middle_point[2] = 0
            max_distance = row[TOP_DOWN_FACE[0]] - row[TOP_DOWN_FACE[1]]
            max_distance[2] = 0
            normalised_distance = np.linalg.norm(max_distance)
            for idx in self.selected_vertex_array:
                landmark = row[idx]
                landmark_copy = np.copy(landmark)
                landmark_copy[2] = 0
                distance = np.linalg.norm(landmark_copy - middle_point)
                new_row.append(distance)
            distance_X.loc[i] = [distance / normalised_distance for distance in new_row]  # type: ignore
        return distance_X
