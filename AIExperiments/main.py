import numpy as np
import dataGenerator
from matplotlib import pyplot as plt
import ml

#Lin reg
'''
data = dataGenerator.genLinRegData(20, 1, 1, noise=0.1)
x = np.linspace(1, 100, 1000)
y = x + 1
fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1], marker='o')
plt.plot(x, y, 'r')
m, b = linearRegression.linear_regression(data[:,0], data[:,1])
y2 = m * x + b
plt.plot(x, y2, 'b')
plt.show()
'''
# KNN

data = dataGenerator.genKnnData(50, 4, noise=0.15)
test_points = []
for i in range(50):
    test_points.append([np.random.rand()*29+1, np.random.rand()*29+1])
test_points = np.array(test_points)
output = ml.KNN(data, test_points)
fig = plt.figure()
plt.scatter(data[0, :, 0], data[0, :, 1], c='r')
plt.scatter(data[1, :, 0], data[1, :, 1], c='g')
plt.scatter(data[2, :, 0], data[2, :, 1], c='b')
plt.scatter(data[3, :, 0], data[3, :, 1], c='y')
plt.scatter(test_points[:, 0], test_points[:, 1], c='black')
fig2 = plt.figure()
c0, c1, c2, c3 = [], [], [], []
for i in range(len(output)):
    if output[i]==0:
        c0.append(test_points[i])
    elif output[i]==1:
        c1.append(test_points[i])
    elif output[i]==2:
        c2.append(test_points[i])
    else:
        c3.append(test_points[i])
c0, c1, c2, c3 = np.array(c0), np.array(c1), np.array(c2), np.array(c3)
plt.scatter(data[0, :, 0], data[0, :, 1], c='r')
if len(c0) > 0:
    plt.scatter(c0[:, 0], c0[:, 1], c='r')
plt.scatter(data[1, :, 0], data[1, :, 1], c='g')
if len(c1) > 0:
    plt.scatter(c1[:, 0], c1[:, 1], c='g')
plt.scatter(data[2, :, 0], data[2, :, 1], c='b')
if len(c2) > 0:
    plt.scatter(c2[:, 0], c2[:, 1], c='b')
plt.scatter(data[3, :, 0], data[3, :, 1], c='y')
if len(c3) > 0:
    plt.scatter(c3[:, 0], c3[:, 1], c='y')
plt.show()


#ANN
## XOR
'''
data = dataGenerator.getXORData(50, noise=0.15)
test_points = []
for i in range(30):
    test_points.append([np.random.rand(), np.random.rand()])
test_points = np.array(test_points)
fig = plt.figure()
plt.scatter(data[0, :, 0], data[0, :, 1], c='r')
plt.scatter(data[1, :, 0], data[1, :, 1], c='b')
plt.scatter(test_points[:, 0], test_points[:, 1], c='black')
plt.show()
'''
'''
data = dataGenerator.getAnnData(50, 4, noise=0.15)
test_points = []
for i in range(50):
    test_points.append([np.random.rand()*29+1, np.random.rand()*29+1])
test_points = np.array(test_points)
#output = ml.KNN(data, test_points)
fig = plt.figure()
plt.scatter(data[0, :, 0], data[0, :, 1], c='r')
plt.scatter(data[1, :, 0], data[1, :, 1], c='g')
plt.scatter(data[2, :, 0], data[2, :, 1], c='b')
plt.scatter(data[3, :, 0], data[3, :, 1], c='y')
plt.scatter(test_points[:, 0], test_points[:, 1], c='black')
plt.show()
'''