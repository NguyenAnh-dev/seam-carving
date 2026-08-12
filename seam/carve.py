import numpy as np
from seam.energy import energy

def find_seam(cost):
    row, col = cost.shape
    dp = cost[0].astype(np.float64)
    back = np.zeros((row, col), dtype=np.int8)

    for i in range(1, row):
        left  = np.concatenate(([np.inf], dp[:-1]))
        mid   = dp
        right = np.concatenate((dp[1:], [np.inf]))

        cand = np.stack([left, mid, right])
        choice = np.argmin(cand, axis=0)

        dp = cand[choice, np.arange(col)] + cost[i]
        back[i] = choice - 1

    idx = int(np.argmin(dp))
    seam = np.empty(row, dtype=np.int32)
    for i in range(row - 1, 0, -1):
        seam[i] = idx
        idx = idx + int(back[i, idx])
    seam[0] = idx
    return seam

def remove_seam(img, seam):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return img[mask].reshape(h, w - 1, img.shape[2])

def carve(img, num_seams):
    for _ in range(num_seams):
        cost = energy(img)
        seam = find_seam(cost)
        img = remove_seam(img, seam)
    return img



def find_seam_logic(cost):
    row = len(cost)
    col = len(cost[0])
    dp = [[0 for _ in range(col)] for _ in range(row)]
    for i in range(col):
        dp[0][i] = cost[0][i]
    for i in range(1, row):
        for j in range(col):
            if j == 0:
                dp[i][0] = min(dp[i-1][0], dp[i-1][1]) + cost[i][0]
            elif j == col - 1:
                dp[i][col-1] = min(dp[i-1][col-2], dp[i-1][col-1]) + cost[i][col-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i-1][j+1]) + cost[i][j]

    mi = float('inf')
    idx = 0
    for j in range(col):
        if dp[row-1][j] < mi:
            mi = dp[row-1][j]
            idx = j
    trace = []
    trace.append(idx)
    n = row-1
    while n != 0:
        if idx != 0 and dp[n-1][idx-1] == dp[n][idx] - cost[n][idx]:
            trace.append(idx-1)
            idx -= 1
            n -= 1
        elif dp[n-1][idx] == dp[n][idx] - cost[n][idx]:
            trace.append(idx)
            n -= 1
        else:
            trace.append(idx+1)
            n -= 1
            idx += 1
    return trace[::-1]