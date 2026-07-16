#!/usr/bin/env python3
"""Module that calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Calculate the shape of a matrix as a list of integers."""
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
