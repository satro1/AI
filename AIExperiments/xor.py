import numpy as np

class Node:
    def __init__(self, weights, to):
        self.val = None
        self.weights = weights
        self.to = to

class NN:
    '''
    layer = list of nodes
    node = value and weights and to nodes
    '''
    def __init__(self, input_nodes, output_nodes):
        layer = []
        self.output_nodes = output_nodes
        for _ in range(output_nodes):
            layer.append(Node(1, None))
        _layer = []
        for _ in range(input_nodes):
            n = Node([], [])
            for i in range(output_nodes):
                n.weights.append(np.random.rand())
                n.to.append(layer[i])
            _layer.append(n)

        self.layers = [_layer, layer]

    def add_layer(self, nodes, fully_connected=True):
        if fully_connected:
            output_layer = self.layers[-1]
            layer = []
            for _ in range(nodes):
                n = Node([], [])
                for i in range(self.output_nodes):
                    n.weights.append(np.random.rand())
                    n.to.append(output_layer[i])
                layer.append(n)



    # Assumption f(x[i]) = y[i] (indicies match)
    def train(self, x, y):
        pass



epochs = 10

train_x = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
])
train_y = np.array([0, 1, 1, 0])

nn = nn(2, 1)