#%%
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import struct
import pickle
from utils import *
#%%
def load_mnist_images(filename):
    """Lê o arquivo de imagens MNIST e retorna um array NumPy com as imagens."""
    with open(filename, 'rb') as f:
        # Ler o cabeçalho do arquivo: primeiro 16 bytes
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Ler o restante dos dados e converter para numpy
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape para (n_imagens, altura, largura)
        images = images.reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    """Lê o arquivo de rótulos MNIST e retorna um array NumPy com os labels."""
    with open(filename, 'rb') as f:
        # Ler o cabeçalho do arquivo: primeiro 8 bytes
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Ler os rótulos restantes e converter para numpy
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Exemplo de uso:
train_images = load_mnist_images('./mnist/train-images.idx3-ubyte')
train_labels = load_mnist_labels('./mnist/train-labels.idx1-ubyte')
test_images = load_mnist_images('./mnist/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('./mnist/t10k-labels.idx1-ubyte')

print(f"Imagens de treino: {train_images.shape}")  # Ex.: (60000, 28, 28)
print(f"Rótulos de treino: {train_labels.shape}")
print(f"Imagens de teste: {test_images.shape}")# Ex.: (60000,)
print(f"Rótulos de test: {test_labels.shape}")

#%%
X = train_images
X = X.reshape(-1,28*28)
Y = train_labels
Y = Y.reshape(-1,1)
X_test = test_images
X_test = X_test.reshape(-1,28*28)
Y_test = test_labels
Y_test = Y_test.reshape(-1,1)
#%%
model2 = Dense_Neural_Diy(input_size=784, hidden_layer1_size=50, hidden_layer2_size=20 , output_size=10)
# %%
model2.weights
# %%
model2.fit(X,Y, learning_rate=0.005, epochs=1, batch_size=40)
# %%
Y_pred_test = model2.predict(X_test).reshape(-1,1)
Y_hat = model2.predict(X).reshape(-1,1).reshape(-1,1)
#%%
np.mean(Y_test == Y_pred_test)
# %%
model2.improve_train(X,Y, learning_rate=0.005, epochs=60, batch_size=40)
# %%
Y_pred_test = model2.predict(X_test).reshape(-1,1)
Y_hat = model2.predict(X).reshape(-1,1).reshape(-1,1)
# %%
np.mean(Y_test == Y_pred_test)
# %%
