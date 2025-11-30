import numpy as np
class NeuralNetwork:
    def __ini__ (self, hidden_layer_size = 100): #Nodi interni inizializzati all'interno del costruttore
        self.hidden_layer_size = hidden_layer_size

    def _accuracy(self, y, y_pred): #frequenza di errore
        return np.sum(y==y_pred)/len(y)

    def _log_loss(self, y_true, y_proba): #Funzione entropica, piu precisa dell'accuracy. Calcola il prodotto scalare.
        return -np.sum(np.dot(y_true,np.log(y_proba))+np.dot((1-y_true),np.log(1-y_proba)))/len(y_true)

    def _relu(self, Z): #Funzione di attivazione per strati nascosti
       return np.maximum(Z, 0)
    
    def _sigmoid(self, Z): #Funzione di attivazione per strati output
       return 1/(1+np.power(np.e,-Z))

    def _forward_propagation(self , X):
        Z1 = np.dot(X,self._W1)+self._b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1,self._W2)+self._b2
        A2 = self._sigmoid(Z2)
        self.foward_cache = (Z1, A1, Z2, A2)
  
  # usiamo il metodo .ravel()
  # per convertire A2 in un array 1D
        return A2.ravel()

    def predict(self, X): #Classifico le probabilita maggiore di 50 come positive dalla foward propagation
        proba = self._forward_propagation(X)
        y = np.zeros(X.shape[0])
        y[proba>=0.5]=1
        y[proba<0.5]=0
        return y

    def predict_proba(self, X):         
       return self._forward_propagation(X)

