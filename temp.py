import numpy as np

def pad_to_4x4(matrix):
    """Pad a 2x3 matrix to 4x4 with zeros"""
    padded = np.zeros((4, 4))
    padded[:2, :3] = matrix
    return padded

def unpad(matrix, rows, cols):
    """Extract the original rows x cols region from a 4x4 matrix"""
    return matrix[:rows, :cols]

# Original 2x3 matrix
original = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float)

# Create two 4x4 matrices for operations
matrix_a = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=float)

matrix_b = np.array([
    [2, -1, 0, 0],
    [0, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 0, 0]
], dtype=float)

print("Original 2x3 matrix:")
print(original)

# Pad the original
padded_original = pad_to_4x4(original)
print("\nPadded to 4x4:")
print(padded_original)

# Perform operations with 4x4 matrices
# Example: (padded_original + matrix_a) * matrix_b
result = (padded_original + matrix_a) @ matrix_b @ matrix_b
print("\nAfter operations (4x4):")
print(result)

# Unpad back to 2x3
unpadded = unpad(result, 2, 2)
print("\nUnpadded back to 2x3:")
print(unpadded)

# Compare with doing operations on original directly
# We need to extract the relevant parts from matrix_a and matrix_b
# For element-wise operations, only positions that overlap with original matter
original_result = (original + matrix_a[:2, :3]) @ matrix_b[:3, :2] @ matrix_b[:2, :2]
print("\nIf operations done on original 2x3 directly:")
print(original_result)

print("\nAre they equal?", np.allclose(unpadded, original_result))