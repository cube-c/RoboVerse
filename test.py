import numpy as np
from scipy.spatial.transform import Rotation as R

# Original vector (example)
v = np.array([1,2,3])  # Replace with your input vector

# Step 1: Define rotation quaternion (180° about Z axis)
q = [0, 0, 1, 0]  # [x, y, z, w]

# Step 2: Define translation
t = np.array([1.2, 0.0, 0.0])

# Step 3: Rotate the vector
rot = R.from_quat(q)
print(rot.as_matrix())
v_rotated = rot.apply(v)
print(v_rotated)

# Step 4: Apply translation
v_transformed = v_rotated + t

print("Transformed vector:", v_transformed)

from scipy.spatial.transform import Rotation as R
import numpy as np

# 입력 쿼터니언 (예: 아무거나)
q_in = [0.0, 0.0, 0.0, 1.0]  # 단위 쿼터니언 (회전 없음)
q = [0, 0, 1, 0]
print(R.from_quat(q).as_matrix())

# XY 평면 기준 90도 회전 = Z축 90도 회전
q_90 = R.from_euler('z', 90, degrees=True).as_quat()
print(q_90, R.from_quat(q_90).as_matrix())

print(f"output: {(R.from_quat(q_90) * R.from_quat(q_in)).as_quat()}")
