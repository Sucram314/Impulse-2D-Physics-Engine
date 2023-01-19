import numpy as np
import numpy.typing as npt
import random

def generateConvex(c,n: int) -> npt.NDArray[np.float64]:
    X_rand, Y_rand = np.sort(np.random.random(n)*c), np.sort(np.random.random(n)*c)
    X_new, Y_new = np.zeros(n), np.zeros(n)

    last_true = last_false = 0
    for i in range(1, n):
        if i != n - 1:
            if random.getrandbits(1):
                X_new[i] = X_rand[i] - X_rand[last_true]
                Y_new[i] = Y_rand[i] - Y_rand[last_true]
                last_true = i
            else:
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]
                last_false = i
        else:
            X_new[0] = X_rand[i] - X_rand[last_true]
            Y_new[0] = Y_rand[i] - Y_rand[last_true]
            X_new[i] = X_rand[last_false] - X_rand[i]
            Y_new[i] = Y_rand[last_false] - Y_rand[i]

    np.random.shuffle(Y_new)
    vertices = np.stack((X_new, Y_new), axis=-1)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

    vertices = np.cumsum(vertices, axis=0)

    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
    vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max

    return vertices

convex = generateConvex(300,20)
string = "["
for point in convex:
    string = string+"["+str(point[0])+","+str(point[1])+"],"

string = string[:-1]
string = string+"]"

print(string)
