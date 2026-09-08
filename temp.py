import numpy as np
from desilofhe import GLEngine

def pad_to_4x4(matrix):
    """Pad a 2x3 matrix to 4x4 with zeros"""
    padded = np.zeros((4, 4))
    padded[:2, :3] = matrix
    return padded

def unpad(matrix, rows, cols):
    """Extract the original rows x cols region from a 4x4 matrix"""
    return matrix[:rows, :cols]

# # Original 2x3 matrix
# original = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ], dtype=float)

# # Create two 4x4 matrices for operations
# matrix_a = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0]
# ], dtype=float)

# matrix_b = np.array([
#     [2, -1, 0, 0],
#     [0, 2, 0, 0],
#     [3, 4, 0, 0],
#     [0, 0, 0, 0]
# ], dtype=float)

# print("Original 2x3 matrix:")
# print(original)

# # Pad the original
# padded_original = pad_to_4x4(original)
# print("\nPadded to 4x4:")
# print(padded_original)

# # Perform operations with 4x4 matrices
# # Example: (padded_original + matrix_a) * matrix_b
# result = (padded_original + matrix_a) @ matrix_b @ matrix_b
# print("\nAfter operations (4x4):")
# print(result)

# # Unpad back to 2x3
# unpadded = unpad(result, 2, 2)
# print("\nUnpadded back to 2x3:")
# print(unpadded)

# # Compare with doing operations on original directly
# # We need to extract the relevant parts from matrix_a and matrix_b
# # For element-wise operations, only positions that overlap with original matter
# original_result = (original + matrix_a[:2, :3]) @ matrix_b[:3, :2] @ matrix_b[:2, :2]
# print("\nIf operations done on original 2x3 directly:")
# print(original_result)

# print("\nAre they equal?", np.allclose(unpadded, original_result))

SUPPORTED_GL_SHAPES = frozenset(
    {
        (256, 16, 16),
        (16, 256, 256),
        (4, 1024, 1024),
        (256, 32, 32),
        (16, 512, 512),
        (4, 2048, 2048),
        (256, 64, 64),
    }
)

for shape in SUPPORTED_GL_SHAPES:
    e = GLEngine(shape=shape)
    print(f"Engine with shape {shape} and max level {e.max_level}")
e = GLEngine(shape=(256, 16, 16))
x = np.random.rand(24, 16)
print(f'original first column: {x[:, 0]}')
m = e.encode(x, level=e.max_level)
print(f"Engine {e.shape} encoded to level {e.max_level}")
decoded = e.decode(m)
print(f"Decoded shape: {decoded.shape}, original shape: {x.shape}")
print(f"Decoded first column: {decoded[0, :, 0].round(8)}")
print(f"Decoded second slot first column: {decoded[1, :, 0].round(8)}")

"""OUTPUT:
Engine with shape (256, 64, 64) and max level 16
Engine with shape (4, 2048, 2048) and max level 6
Engine with shape (256, 32, 32) and max level 6
Engine with shape (16, 256, 256) and max level 1
Engine with shape (256, 16, 16) and max level 1
Engine with shape (4, 1024, 1024) and max level 1
Engine with shape (16, 512, 512) and max level 6
original first column: [0.67909773 0.44451688 0.57950627 0.78709001 0.00768431 0.39224605
 0.2344535  0.70773615 0.4656473  0.33842366 0.79125725 0.47431794
 0.80790451 0.65318135 0.84267619 0.80441119 0.25067933 0.84149384
 0.35119676 0.0854395  0.32402255 0.89731419 0.63950169 0.46589453]
Engine (256, 16, 16) encoded to level 1
Decoded shape: (256, 16, 16), original shape: (24, 16)
Decoded first column: [0.67909773 0.44451688 0.57950627 0.78709001 0.00768431 0.39224605
 0.2344535  0.70773615 0.4656473  0.33842366 0.79125725 0.47431794
 0.80790451 0.65318135 0.84267619 0.80441119]
Decoded second slot first column: [ 0.25067933  0.84149384  0.35119676  0.0854395   0.32402255  0.89731419
  0.63950169  0.46589453  0.          0.         -0.         -0.
  0.          0.         -0.          0.        ]
"""