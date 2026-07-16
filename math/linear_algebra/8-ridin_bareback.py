#!/usr/bin/env python3
"""Module that performs matrix multiplication."""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication of mat1 and mat2."""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for col in range(len(mat2[0])):
            total = 0
            for i in range(len(mat2)):
                total += row[i] * mat2[i][col]
            new_row.append(total)
        result.append(new_row)
    return result
