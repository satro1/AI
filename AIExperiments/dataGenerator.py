import numpy as np

def genLinRegData(n, slope, intercept, min=1, max=100, noise=0.5):
    if min > max:
        raise Exception("Invalid: Min > Max")
    r = max - min
    arr = []
    for i in range(n):
        x = np.random.rand() * r + min
        y = (slope  * x + intercept) + ((-1) ** np.random.randint(0, 2)) * r * noise * np.random.rand()
        arr.append([x, y])
    return np.array(arr)

def genKnnData(n_per_class, n_classes, x=(1,30), y=(1,30), noise=0.5):
    if x[0] > x[1] or y[0] > y[1]:
        raise Exception("Invalid: Min > Max")
    classes = []
    rx = x[1] - x[0]
    ry = y[1] - y[0]
    for i in range(n_classes):
        center = [np.random.rand() * rx + x[0], np.random.rand() * ry + y[0]]
        classes.append([center])
    for i in range(len(classes)):
        for j in range(n_per_class):
            center = classes[i][0]
            x_new = center[0] + ((-1) ** np.random.randint(0, 2)) * rx * noise * np.random.rand()
            y_new = center[1] + ((-1) ** np.random.randint(0, 2)) * ry * noise * np.random.rand()
            classes[i].append([x_new, y_new])
    return np.array(classes)

def getXORData(n_per_class, noise=0.5):
    classes = [[[0.25, 0.25], [0.75, 0.75]], [[0.25, 0.75], [0.75, 0.25]]]
    for i in range(2):
        for j in range(n_per_class):
            center = classes[i][np.random.randint(0, 2)]
            x_new = center[0] + ((-1) ** np.random.randint(0, 2)) * noise * np.random.rand()
            y_new = center[1] + ((-1) ** np.random.randint(0, 2)) * noise * np.random.rand()
            classes[i].append([x_new, y_new])
    return np.array(classes)


def getAnnData(n_per_class, n_classes, x=(1,30), y=(1,30), noise=0.5):
    return genKnnData(n_per_class, n_classes, x, y, noise)
