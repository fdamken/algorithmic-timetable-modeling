import os
from dataclasses import dataclass

import numpy as np
import wandb


IS_EXPERIMENT = os.environ.get('IS_EXPERIMENT') is not None
if IS_EXPERIMENT:
    wandb.init(project = 'amf')



@dataclass
class EventNetworkEdge:
    source: str
    target: str
    weight: float
    lower_bound: int
    upper_bound: int


    def basic_edge(self):
        return self.source, self.target



class NamedVector:
    def __init__(self, keys):
        self._vector = np.zeros((len(keys), 1))

        self._indices = { }
        for i, key in enumerate(keys):
            self._set_index(key, i)


    def _construct_named_index(self, key):
        raise NotImplementedError('Abstract method _construct_named_index(self, key) has to be implemented!')


    def __repr__(self):
        return "_indices: %s; _vector: %s" % (repr(self._indices), repr(self._vector))


    def __str__(self):
        if self._vector.shape[1] == 1:
            return str(self._vector.T) + '^T'
        return str(self._vector)


    def get_sub_vector(self, keys):
        result = []
        for key in keys:
            result.append(self.get_named_value(key))
        return np.array(result).reshape(-1, 1)


    def get_vector(self):
        return self._vector


    def set_vector(self, vector):
        self._vector = vector


    def get_named_value(self, key):
        return self._vector[self.get_index(key)]


    def set_named_value(self, key, value):
        self._vector[self.get_index(key)] = value


    def get_index(self, key):
        return self._indices[self._construct_named_index(key)]


    def _set_index(self, key, index):
        self._indices[self._construct_named_index(key)] = index



class NodeVector(NamedVector):
    def _construct_named_index(self, key):
        return key



class EdgeVector(NamedVector):
    def _construct_named_index(self, key):
        if isinstance(key, EventNetworkEdge):
            key = key.basic_edge()
        return str(sorted([key[0], key[1]]))



class NamedMatrix:
    def __init__(self, row_keys, col_keys):
        self._matrix = np.zeros((len(row_keys), len(col_keys)))

        self._row_indices = { }
        self._col_indices = { }
        for i, row_key in enumerate(row_keys):
            self._set_row_index(row_key, i)
        for i, col_key in enumerate(col_keys):
            self._set_col_index(col_key, i)


    def _construct_named_row_index(self, key):
        raise NotImplementedError('Abstract method _construct_named_row_index(self, key) has to be implemented!')


    def _construct_named_col_index(self, key):
        raise NotImplementedError('Abstract method _construct_named_col_index(self, key) has to be implemented!')


    def __repr__(self):
        return "_row_indices: %s; _col_indices: %s; _matrix: %s" % (
                str(self._row_indices), str(self._col_indices), str(self._matrix))


    def __str__(self):
        return str(self._matrix)


    def get_matrix(self):
        return self._matrix


    def set_matrix(self, matrix):
        self._matrix = matrix


    def get_named_value(self, row_key, col_key):
        return self._matrix[self.get_row_index(row_key), self.get_col_index(col_key)]


    def set_named_value(self, row_key, col_key, value):
        self._matrix[self.get_row_index(row_key), self.get_col_index(col_key)] = value


    def get_named_row(self, row_key):
        return self._matrix[self.get_row_index(row_key), :]


    def set_named_row(self, row_key, col_vector):
        self._matrix[self.get_row_index(row_key), :] = col_vector


    def get_named_col(self, col_key):
        return self._matrix[:, self.get_col_index(col_key)]


    def set_named_col(self, col_key, row_vector):
        self._matrix[:, self.get_col_index(col_key)] = row_vector


    def get_row_index(self, key):
        return self._row_indices[self._construct_named_row_index(key)]


    def get_col_index(self, key):
        return self._col_indices[self._construct_named_col_index(key)]


    def _set_row_index(self, key, index):
        self._row_indices[self._construct_named_row_index(key)] = index


    def _set_col_index(self, key, index):
        self._col_indices[self._construct_named_col_index(key)] = index



class EdgeMatrix(NamedMatrix):
    def _construct_named_row_index(self, key):
        return self._construct_named_index(key)


    def _construct_named_col_index(self, key):
        return self._construct_named_index(key)


    def _construct_named_index(self, key):
        return str(sorted([key[0], key[1]]))
