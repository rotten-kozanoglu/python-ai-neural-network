import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# veri kümesi oluştur.
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

# veri setini dizi ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros((y.shape[0],1)) #çıkış şeklini düzelt

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        # ağırlıklara göre kayıp fonksiyonunun türevini bulmak için zincir kuralının uygulanması..
        # d_weights1 issue. -- 
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (2*(self.y - self.output) * sigmoid_derivative(self.output))).dot(self.weights2.T).dot(sigmoid_derivative(self.layer1))
        # -- 

        self.weights1 += d_weights1
        self.weights2 += d_weights2

learning_rate = 0.1
num_epochs = 100

nn = NeuralNetwork(X_train, y_train)

for epoch in range(num_epochs):
 nn.feedforward()
 nn.backprop()

predictions = nn.feedforward()
predictions = predictions > 0.5

accuracy = accuracy_score(y_test, predictions)
print("Kesinlik:", accuracy) 


