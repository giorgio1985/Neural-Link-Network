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

    def fit(self, X, y, epochs=200, lr=0.01): #Gradient descent
     for _ in range(epochs):
      Y = self._forward_propagation(X)
      dW1, db1, dW2, db2 = self._funzione_magica_che_calcola_le_derivate_parziali(X, y)
      self._W1-=lr*dW1
      self._b1-=lr*db1
      self._W2-=lr*dW2


    def _relu_derivative(self, Z):

      dZ = np.zeros(Z.shape)
      dZ[Z>0] = 1
      return dZ


    def _back_propagation(self, X, y):
      Z1, A1, Z2, A2 = self._forward_cache
                   
      m = A1.shape[1]
    
      dZ2 = A2-y.reshape(-1,1) # il reshape ci serve per far combaciare le dimensioni dei due vettori
      dW2 = np.dot(A1.T, dZ2)/m
      db2 = np.sum(dZ2, axis=0)/m

      dZ1 = np.dot(dZ2, self._W2.T)*self._relu_derivative(Z1)
      dW1 = np.dot(X.T, dZ1)/m
      db1 = np.sum(dZ1, axis=0)/m # eseguiamo la somma lungo le righe
    
      return dW1, db1, dW2, db2
