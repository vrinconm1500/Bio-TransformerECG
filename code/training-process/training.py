# Author: Victor Rincon 
# ECG Biometric Indetification & Authentication using Transformer and Transformer-Siamese 

# Importamos librerias 
import utils
import transformerECG

#Cargamos el dataset
path_dataset = "Codigos/data/ready/pickles/ecgid_290.pickle"
y, x, people = utils.load_data(path_dataset)
y, x = utils.shuffle_data(y, x)

#Definimos el input shape con la dimension de la señal
SIG_DIMS = (x.shape[1], 1)
print("Input Shape:", SIG_DIMS, "\n")

x_train, y_train, x_valid, y_valid, x_test, y_test, lb = utils.split_data(y, x, SIG_DIMS)
data = x_train, y_train, x_valid, y_valid, x_test, y_test, lb


# Definir los parámetros de entrenamiento
folder = 'modelos/'  # Carpeta donde se guardarán los resultados y el modelo
epochs = 2  # Número de épocas de entrenamiento
initial_lr = 0.003  # Tasa de aprendizaje inicial
decay_rate = 0.5  # Tasa de reducción del learning rate
decay_epochs = [30, 60]  # Épocas en las que se reducirá el learning rate

model, history = transformerECG.train(folder, SIG_DIMS, data, epochs, initial_lr, decay_rate, decay_epochs)