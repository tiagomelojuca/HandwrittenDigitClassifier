import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import pickle
from utils import *
from dense_neural_class import *

# Função para carregar o modelo com caminho absoluto
def load_model(filename):
    # Pega o diretório atual onde o script está sendo executado
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Constrói o caminho completo do arquivo .pkl
    filepath = os.path.join(current_dir, filename + '.pkl')
    
    with open(filepath, 'rb') as file:
        model_loaded = pickle.load(file)
    
    return model_loaded

# Carrega o modelo ao iniciar o programa
model = load_model('model')

def predict(vetor):
    # Usa o modelo carregado para fazer a predição
    resultado = model.predict(vetor)[0]
    messagebox.showinfo("Result", f"The number is : {resultado}")

# Classe do aplicativo de desenho
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing Canvas 28x28")

        # Configurações do canvas
        self.canvas_size = 280  # Tamanho do canvas em pixels
        self.image_size = 28  # Tamanho da imagem para vetorizar
        self.brush_size = 10  # Tamanho do pincel branco

        # Canvas para desenhar
        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Criação da imagem e do objeto para desenhar
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Botões de ação
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.predict_button = tk.Button(self.button_frame, text="Tell the number", command=self.predict_image)
        self.predict_button.pack(side="left")

        self.clear_button = tk.Button(self.button_frame, text="Erase", command=self.clear_canvas)
        self.clear_button.pack(side="left")

        # Evento de desenho
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Desenhar na tela e na imagem
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
        # Desenha no canvas (tela) com pincel branco
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

        # Desenha na imagem de 28x28 para vetorização
        scaled_x1, scaled_y1 = (x1 * self.image_size // self.canvas_size), (y1 * self.image_size // self.canvas_size)
        scaled_x2, scaled_y2 = (x2 * self.image_size // self.canvas_size), (y2 * self.image_size // self.canvas_size)
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="white")

    def predict_image(self):
        # Converter a imagem para um vetor e normalizar os valores (0 a 1)
        image_data = np.array(self.image).reshape(1, -1) / 255.0
        predict(image_data)

    def clear_canvas(self):
        # Limpa o canvas e recria uma nova imagem preta
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

# Inicialização do aplicativo
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
