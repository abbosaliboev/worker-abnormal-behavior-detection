"""
COCO 17-joint skeleton graph for ST-GCN (YOLO-pose output format).

Joint indices (COCO):
  0:nose
  1:left_eye   2:right_eye
  3:left_ear   4:right_ear
  5:left_shoulder  6:right_shoulder
  7:left_elbow     8:right_elbow
  9:left_wrist    10:right_wrist
 11:left_hip      12:right_hip
 13:left_knee     14:right_knee
 15:left_ankle    16:right_ankle
"""

import numpy as np

NUM_JOINTS = 17

EDGES = [
    # head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # left arm
    (5, 7), (7, 9),
    # right arm
    (6, 8), (8, 10),
    # left leg
    (11, 13), (13, 15),
    # right leg
    (12, 14), (14, 16),
]

# center of gravity — mid-hip (use left_hip as BFS root)
CENTER = 11


def _build_adjacency():
    """Returns A of shape (3, V, V) — spatial configuration partitioning."""
    V = NUM_JOINTS

    adj = {i: set() for i in range(V)}
    for a, b in EDGES:
        adj[a].add(b)
        adj[b].add(a)

    # BFS distance from center
    dist = {CENTER: 0}
    queue = [CENTER]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    for i in range(V):
        if i not in dist:
            dist[i] = 0

    A = np.zeros((3, V, V), dtype=np.float32)

    # subset 0: self-link
    for i in range(V):
        A[0, i, i] = 1

    # subsets 1 (centripetal) & 2 (centrifugal)
    for a, b in EDGES:
        if dist[a] < dist[b]:
            A[1, b, a] = 1
            A[2, a, b] = 1
        elif dist[a] > dist[b]:
            A[1, a, b] = 1
            A[2, b, a] = 1
        else:
            A[1, a, b] = 1
            A[1, b, a] = 1

    # row-normalise
    for k in range(3):
        row_sum = A[k].sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        A[k] /= row_sum

    return A


ADJACENCY = _build_adjacency()
