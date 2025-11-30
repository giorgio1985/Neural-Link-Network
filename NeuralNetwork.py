import numpy as np
class NeuralNetwork:
    def __ini__ (self, hidden_layer_size = 100): # Nodi interni inizializzati all'interno del costruttore
        self.hidden_layer_size = hidden_layer_size

    def _accuracy(self, y, y_pred):   # frequenza di errore
        return np.sum(y==y_pred)/len(y)

    def _log_loss(self, y_true, y_proba):  # Funzione entropica, piu precisa dell'accuracy. Calcola il prodotto scalare.
        return -np.sum(np.dot(y_true,np.log(y_proba))+np.dot((1-y_true),np.log(1-y_proba)))/len(y_true)
    






