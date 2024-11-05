import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_vectorized(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    
    return np.exp(x)/np.sum(np.exp(x),axis = 1,keepdims=True)

def cost(A3, Y):
    m = Y.shape[0]
    return -np.sum(Y * np.log(A3 + 1e-8)) / m
    
def one_hot(y):
    n_classes = np.max(y)+1
    y_hot = np.zeros((y.shape[0],n_classes))
    for n in range(y.shape[0]):
        y_hot[n][y[n][0]]=1
    
    return y_hot

def random_weights(input_size, hidden_layer1_size, hidden_layer2_size, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_layer1_size) * 0.01,
        'b1': np.zeros((1, hidden_layer1_size)),
        'W2': np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01,
        'b2': np.zeros((1, hidden_layer2_size)),
        'W3': np.random.randn(hidden_layer2_size, output_size) * 0.01,
        'b3': np.zeros((1, output_size)),
    }
    return weights

def forward(X, weights, return_back='forward'):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = relu_vectorized(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = relu_vectorized(Z2)
    Z3 = np.dot(A2, weights['W3']) + weights['b3']
    A3 = softmax(Z3)
    forward_data = (Z1, A1, Z2, A2, Z3, A3)
    if return_back == 'forward': 
        return forward_data
    elif return_back == 'predict':
        return A3

def back_propagation(X, Y, forward_data, weights):
    Z1, A1, Z2, A2, Z3, A3 = forward_data
    m = X.shape[0]

    dZ3 = A3 - Y
    gradient_W3 = np.dot(A2.T, dZ3) / m
    gradient_b3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, weights['W3'].T)
    dZ2 = dA2 * relu_derivative(Z2)
    gradient_W2 = np.dot(A1.T, dZ2) / m
    gradient_b2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * relu_derivative(Z1)
    gradient_W1 = np.dot(X.T, dZ1) / m
    gradient_b1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        'gradient_W3': gradient_W3, 'gradient_b3': gradient_b3,
        'gradient_W2': gradient_W2, 'gradient_b2': gradient_b2,
        'gradient_W1': gradient_W1, 'gradient_b1': gradient_b1
    }
    return gradients

def update_weights(weights, gradients, learning_rate):
    weights['W3'] -= gradients['gradient_W3'] * learning_rate
    weights['b3'] -= gradients['gradient_b3'] * learning_rate
    weights['W2'] -= gradients['gradient_W2'] * learning_rate
    weights['b2'] -= gradients['gradient_b2'] * learning_rate
    weights['W1'] -= gradients['gradient_W1'] * learning_rate
    weights['b1'] -= gradients['gradient_b1'] * learning_rate

    return weights

def train(X, Y, input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate, epochs, batch_size, weights):
    #weights = random_weights(input_size, hidden_layer1_size, hidden_layer2_size, output_size)
    m = X.shape[0]
    Y = one_hot(Y)

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            forward_data = forward(X_batch, weights)
            gradients = back_propagation(X_batch, Y_batch, forward_data, weights)
            weights = update_weights(weights, gradients, learning_rate)

        # Cálculo do custo total após uma época (opcional)
        forward_data_full = forward(X, weights)
        _, _, _, _, _, A3 = forward_data_full
        epoch_cost = cost(A3, Y)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, cost: {epoch_cost}')
    
    return weights

def predict(X, weights):
    A3 = forward(X, weights, return_back = 'predict')
    return np.argmax(A3, axis=1)

def save_model(name, model):
    with open(name+'.pkl', 'wb') as file: 
        pickle.dump(model,file)

def load_model(name):
    with open(name+'.pkl', 'rb') as file:
        model_loaded = pickle.load(file)

    return model_loaded

