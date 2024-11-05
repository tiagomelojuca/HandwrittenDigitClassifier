from utils import *
class Dense_Neural_Diy:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, output_size):
        self.input_size = input_size
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.output_size = output_size
        self.weights = None
        # init methods
        self.random_weights()

    def random_weights(self):
        # initialize random weights when the object is created
        self.weights = random_weights(self.input_size, self.hidden_layer1_size, self.hidden_layer2_size, self.output_size)

    def fit(self, X,Y,learning_rate, epochs, batch_size):
        self.weights = train(X, Y, self.input_size, self.hidden_layer1_size, self.hidden_layer2_size, self.output_size, learning_rate, epochs, batch_size, self.weights)

    def predict(self,X):
        return predict(X,self.weights)

    def improve_train(self,X,Y,learning_rate, epochs, batch_size):
        self.weights = train(X, Y, self.input_size, self.hidden_layer1_size, self.hidden_layer2_size, self.output_size, learning_rate, epochs, batch_size, self.weights)
        