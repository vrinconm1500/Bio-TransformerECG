import utils
from keras.models import  load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

#Cargamos el dataset
path_dataset = "/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/ready/pickles/CYBHi-1500.pickle"
y, x, people = utils.load_data(path_dataset)
y, x = utils.shuffle_data(y, x)
#/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/ready/pickles
#Definimos el input shape con la dimension de la señal
SIG_DIMS = (x.shape[1], 1)
print("Input Shape:", SIG_DIMS, "\n")

x_train, y_train, x_valid, y_valid, x_test, y_test, lb = utils.split_data(y, x, SIG_DIMS)
data = x_train, y_train, x_valid, y_valid, x_test, y_test, lb


#cargamos el modelo 
path_model = "modelos/CYBHi_1500.h5"
best_model = load_model(path_model)

# evaluate model
_, accuracy = best_model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('\n', 'Test accuracy:', accuracy, '\n')

_,accuracy = best_model.evaluate(x_valid, y_valid, batch_size=64, verbose = 1)
print('\n', 'Valid accuracy:', accuracy, '\n')

_,accuracy = best_model.evaluate(x_train, y_train, batch_size=64, verbose = 1)
print('\n', 'Train accuracy:', accuracy, '\n')

lbb = LabelBinarizer()
predictions = best_model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(predictions, axis=1)
y_pred_bool = lbb.fit_transform(y_pred_bool)
print(classification_report(y_test, y_pred_bool))