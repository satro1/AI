import numpy as np

def linear_regression(x, y):
    A, b = np.ones((len(x), 2)), np.array(y)
    A[:,0] = np.array(x)
    A_t = np.transpose(A)
    output = np.matmul(np.linalg.inv(np.matmul(A_t, A)), np.matmul(A_t, b))
    return output

def KNN(data, points, k = 7):
    if k % 2 == 0:
        raise Warning("Even K might result in ambiguity")
    output = []
    for i in range(len(points)):
        distances = []
        c = []
        for class_ in range(len(data)):
            for point in data[class_]:
                distances.append((point[0] - points[i][0]) ** 2 + (point[1] - points[i][1]) ** 2)
                c.append(class_)
        c = [x for _, x in sorted(zip(distances, c))]
        output.append(max(c, key=c.count))
    return np.array(output)