import numpy as np
class NeuralNetwork:
    def __ini__ (self, hidden_layer_size = 100):
        self.hidden_layer_size = hidden_layer_size

    def _accuracy(self, y, y_pred):
        return np.sum(y==y_pred)/len(y)


    






