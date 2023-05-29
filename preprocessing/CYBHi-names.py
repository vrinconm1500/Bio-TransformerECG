#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:27:34 2023

@author: victorrincon
"""

import os
import glob
import pandas as pd

# Ruta de la carpeta que contiene los datos de CYBHi
folder_path = "/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/raw/CYBHi/data/long-term"

# Crear una lista con todas las rutas de los archivos TXT de la carpeta
file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

# Crear un contador para asignar un nombre único a cada archivo CSV generado
count = 1

# Iterar sobre cada archivo de datos
for file_path in file_paths:
    # Leer el archivo de datos y crear un dataframe de pandas con las columnas separadas por comas
    df = pd.read_csv(file_path, sep=",")
    # Extraer la señal ECG del dataframe (asumiendo que la señal ECG está en la segunda columna)
    ecg_signal = df.iloc[5:, 0]
    # Crear un dataframe con la señal ECG
    ecg_df = pd.DataFrame(ecg_signal, columns=["ECG"])
    # Crear el nombre del archivo CSV
    csv_name = "patient_" + str(count) + ".csv"
    # Guardar el dataframe de la señal ECG como archivo CSV en la misma carpeta que los datos originales
    ecg_df.to_csv(os.path.join(folder_path, csv_name), index=False)
    # Incrementar el contador para asignar un nombre único al siguiente archivo CSV
    count += 1
    
    
from biosppy.signals import ecg
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import matplotlib

df1 = pd.read_csv("data/raw/CYBHi/data/short-term-csv/patient_SM_1.csv", sep=",")
df2 = pd.read_csv("data/raw/CYBHi/data/long-term-csv/patient_RA_2.csv")

filtered1 = df1['3'].values
filtered2 = df2['0'].values
sgs = np.concatenate((sgs, filtered))


ecg_signal = np.concatenate((filtered1, filtered2))
a = np.array(ecg_signal, dtype="float32")

a = (a - a.min()) / (a.max() - a.min())
a = np.array(ecg_signal)

a = list(ecg_signal)


def refine_r_peaks(sig, r_peaks):
    r_peaks2 = np.array(r_peaks)  # make a copy
    for i in range(len(r_peaks)):
        r = r_peaks[i]  # current R-peak
        small_segment = sig[max(0, r - 100):min(len(sig), r + 100)]  # consider the neighboring segment of R-peak
        r_peaks2[i] = np.argmax(small_segment) - 100 + r_peaks[i]  # picking the highest point
        r_peaks2[i] = min(r_peaks2[i], len(sig))  # the detected R-peak shouldn't be outside the signal
        r_peaks2[i] = max(r_peaks2[i], 0)  # checking if it goes before zero
    return r_peaks2  # returning the refined r-peak list


def segment_signals(sig, r_peaks_annot, bmd=True, normalization=True):
    segmented_signals = []
    r_peaks = np.array(r_peaks_annot)
    r_peaks = refine_r_peaks(sig, r_peaks)
    if bmd:
        win_len = 300
    else:
        win_len = 256
    win_len_1_4 = win_len // 4
    win_len_3_4 = 3 * (win_len // 4)
    for r in r_peaks:
        if ((r - win_len_1_4) < 0) or ((r + win_len_3_4) >= len(sig)):  # not enough signal to segment
            continue
        segmented_signal = np.array(sig[r - win_len_1_4:r + win_len_3_4])  # segmenting a heartbeat

        if normalization:  # Z-score normalization
            if abs(np.std(segmented_signal)) < 1e-6:  # flat line ECG, will cause zero division error
                continue
            segmented_signal = (segmented_signal - np.mean(segmented_signal)) / np.std(segmented_signal)

        if not np.isnan(segmented_signal).any():  # checking for nan, this will never happen
            segmented_signals.append(segmented_signal)

    return segmented_signals, r_peaks

#Perform QRS detection
ecgOut = ecg.ecg(signal=filtered1, sampling_rate=1000., show=True,interactive=False)#[4]

peaks = ecgOut["rpeaks"]
wavess= ecgOut["templates"]



waves, pks = segment_signals(filtered1, peaks, False, True)


plt.plot(wavess[25])
length = len(waves)

for k in range(length):
    wave = waves[k]
    plt.title("RA")
    self.augment(wave, len(wave))
    how_many.append(len(wave))
    count += 1

plt.show()
#print("Len per Wave", how_many)
print("Mean per Wave", np.mean(how_many))
print("How many", len(how_many))
print("Total", len(how_many) * 9)





import os
import pandas as pd

data_path = "/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/raw/CYBHi/data/long-term"

nombres = []  #creamos una variable listado vacia para guardar los nombres
numero = []   #creamos una variable listado vacia para guardar el identificador de las lecturas
# Recorre todos los archivos TXT en la carpeta de datos de largo plazo
for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_path, filename)
        ecg_df = pd.read_csv(file_path, sep=",", header=None)
        ecg_df = ecg_df.iloc[6:, 0]
        nombres.append(filename) #guardamos cada nombre de archivo del directo destino en esta variable
        item = filename[9:]
        div = item.split("-",maxsplit=1)
        if div[0]+"_1" in numero:
            numero.append(div[0]+"_2")
            csv_name = "patient_" + div[0]+"_2" + ".csv"
            print(csv_name)
            ecg_df.to_csv(os.path.join(data_path, csv_name), index=False, )
        else:
            
            numero.append(div[0]+"_1")  # guardamos el numero del archivo
            csv_name = "patient_" + div[0]+"_1" + ".csv"
            print(csv_name)
            ecg_df.to_csv(os.path.join(data_path, csv_name), index=False)
            












            
        file_path = os.path.join(data_path, filename)
       
        # Lee el archivo TXT utilizando Pandas y extrae la señal de ECG
        ecg_df = pd.read_csv(file_path, sep=",")
        ecg_df = ecg_df.iloc[5:, 0]
        # Crea un archivo CSV con el DataFrame de la señal de ECG
        csv_name = "patient_" + root_code + ".csv"
        print(csv_name)
        nombres.append(csv_name)

        ecg_df.to_csv(os.path.join(data_path, csv_name), index=False)




import os
import pandas as pd

data_path = "/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/raw/CYBHi/data/long-term"

nombres = []  #creamos una variable listado vacia para guardar los nombres
numero = []   #creamos una variable listado vacia para guardar el identificador de las lecturas

# Recorre todos los archivos TXT en la carpeta de datos de largo plazo
for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        nombres.append(filename) #guardamos cada nombre de archivo del directo destino en esta variable
        item = filename[9:]
        div = item.split("-",maxsplit=1)
        if div[0]+"_1" in numero:
            numero.append(div[0]+"_2")
        else:
            
            numero.append(div[0]+"_1")  # guardamos el numero del archivo
        
        file_path = os.path.join(data_path, filename)
        with open(file_path, "r") as f:
            # Extrae el código raíz del nombre del archivo
            
            item = filename[9:]
            root_code = item.split("-",maxsplit=1)
            # Lee las líneas del archivo y extrae la señal de ECG
            ecg_data = []
            for line in f.readlines():
                ecg_value = line.split("\t")[0]
                ecg_data.append(float(ecg_value))
            # Crea un DataFrame de pandas con la señal de ECG y lo guarda en un archivo CSV
            ecg_df = pd.DataFrame(ecg_data, columns=["ECG"])
            csv_name = "patient_" + root_code + ".csv"
            print(csv_name)
            # ecg_df.to_csv(os.path.join(data_path, csv_name), index=False)








import os
import pandas as pd


patients = os.listdir(data_path)

for patient in patients:
    ecg_files = [f for f in os.listdir(os.path.join(data_path, patient)) if f.endswith(".txt")]
    for i, ecg_file in enumerate(ecg_files):
        ecg_data = pd.read_csv(os.path.join(data_path, patient, ecg_file), header=None)
        output_filename = f"patient_{patient}-{i+1}.csv"
        ecg_data.to_csv(output_filename, index=False)








#Short - TERM 





import os
import pandas as pd

data_path = "raw/CYBHi/data/short-term"

nombres = []  #creamos una variable listado vacia para guardar los nombres
numero = []   #creamos una variable listado vacia para guardar el identificador de las lecturas
# Recorre todos los archivos TXT en la carpeta de datos de largo plazo
for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_path, filename)
        ecg_df = pd.read_csv(file_path, sep="\t", skiprows=8, header=None)[3]
        #ecg_df = ecg_df.iloc[6:, 0]
        nombres.append(filename) #guardamos cada nombre de archivo del directo destino en esta variable
        item = filename[9:]
        div = item.split("-",maxsplit=1)
        
        if div[0]+"_1" in numero:
            numero.append(div[0]+"_2")
            csv_name = "patient_" + div[0]+"_2" + ".csv"
            print(csv_name)
            ecg_df.to_csv(os.path.join(data_path, csv_name), index=False, )
        
        else:
            
            numero.append(div[0]+"_1")  # guardamos el numero del archivo
            csv_name = "patient_" + div[0]+"_1" + ".csv"
            print(csv_name)
            ecg_df.to_csv(os.path.join(data_path, csv_name), index=False)






import os
import pandas as pd

data_path = "raw/CYBHi/data/short-term"  # Especifica la ruta de tus archivos TXT
output_path = "raw/CYBHi/data/short-term-csv"  # Especifica la ruta para guardar los archivos CSV

def procesar_archivo(file_path, paciente, identificador):
    ecg_df = pd.read_csv(file_path, sep="\t", skiprows=8, header=None)[3]
    csv_name = f"patient_{paciente}_{identificador}.csv"
    csv_path = os.path.join(output_path, csv_name)
    ecg_df.to_csv(csv_path, index=False, )
    print(f"Archivo CSV guardado: {csv_name}")

# Obtener la lista de archivos TXT
archivos_txt = [filename for filename in os.listdir(data_path) if filename.endswith(".txt")]
archivos_txt.sort()


identificadores = ["1", "2", "3", "4", "5", "6"]

for i, filename in enumerate(archivos_txt):
    item = filename[9:]  # Obtener las iniciales del paciente del nombre del archivo
    div = item.split("-",maxsplit=1)
    paciente = div[0]
    identificador = identificadores[i % len(identificadores)]  # Asignar el identificador de forma automática
    
    file_path = os.path.join(data_path, filename)
    print(file_path)
    procesar_archivo(file_path, paciente, identificador)


















import numpy as np
unique, counts = np.unique(nombres, return_counts=True)

result = np.column_stack((unique, counts)) 
print (result)



df1 = pd.read_csv("data/raw/CYBHi/data/cybhi-lst-csv/patient_AA_1.csv", names="Hola")

df = pd.read_csv("data/raw/CYBHi/data/short-term/20110715-MLS-A1-8B.txt", sep="\t", skiprows=8, header=None)[3]

serie = df1.squeeze()

try:
    datos = df1["0"].values 
    
except:
    datos = df1["3"].values 


fig = plt.figure(figsize=(15,8))

plt.plot(datos)
plt.xlabel('sample')
plt.xlim(0, 134775)
plt.xticks(np.arange(0, 134775, 1000 * 4))
plt.ylabel('ECG [μV]')
plt.ylim(-2000, 5000)
plt.grid()

plt.show()


#Perform QRS detection
ecgOut = ecg.ecg(signal=datos, sampling_rate=1000., show=True,interactive=False)

peaks = ecgOut["rpeaks"]
wavess= ecgOut["templates"]



waves, pks = segment_signals(a, peaks, False, True)




import biosppy.signals as biosig
import matplotlib.pyplot as plt

# Supongamos que tienes una señal de ECG llamada 'ecg_signal'

# Aplicar filtro a la señal de ECG para mejorarla
filtered_ecg = biosig.tools.filter_signal(datos, ftype='FIR', band='bandpass', order=10, frequency=[3, 45], sampling_rate=1000)[0]

# Graficar la señal original y la señal filtrada
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(datos)
plt.title('Señal de ECG original')
plt.subplot(2, 1, 2)
plt.plot(filtered_ecg)
plt.title('Señal de ECG filtrada')
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd

dataframe = pd.read_csv("data/raw/CYBHi/data/short-term/20110715-MLS-A1-8B.txt", sep="\t", skiprows=8, header=None)




import pandas as pd
import numpy as np
from scipy.signal import firwin, lfilter

def preprocess_signal(ecg_data):
    # Low pass filter
    Fs = 500
    Fstop = 120
    Fpass = 90
    Dstop = 0.0001
    Dpass = 0.05
    N = int(np.ceil((4 / Dstop)))
    b = firwin(N, Fpass, fs=Fs, pass_zero=True)
    ecg_data = lfilter(b, 1, ecg_data)
    
    # Filter out 50 Hz
    Fpass1 = 48
    Fstop = 50
    Fpass2 = 52
    N = int(np.ceil((4 / Dstop)))
    bands = [0, Fpass1, Fstop, Fpass2, Fs / 2]
    desired = [1, 1, 0, 1, 1]
    b = firwin(N, bands, fs=Fs, pass_zero=False, window='hamming', scale='linear')
    ecg_data = lfilter(b, 1, ecg_data)
    
    # High pass filter (remove signal wandering)
    Fstop = 0.1
    Fpass = 0.7
    Dstop = 0.0001
    Dpass = 0.05
    N = int(np.ceil((4 / Dstop)))
    b = firwin(N, Fpass, fs=Fs, pass_zero=False)
    ecg_data = lfilter(b, 1, ecg_data)
    
    return ecg_data

file_path = "data/raw/CYBHi/data/short-term/20110715-MLS-A1-8B.txt"

data = pd.read_csv("data/raw/CYBHi/data/short-term/20110715-MLS-A1-8B.txt", sep="\t", skiprows=8, header=None)[3]
data = data.values


filtered_signal = preprocess_signal(data)


new_data = np.zeros((5000, 12))
for j in range(12):
    ecg_signal = data[:, j]
    filtered_signal = preprocess_signal(ecg_signal)
    new_data[:, j] = filtered_signal[1001:6001]

output_file_path = "path/to/your/output/preprocessed_file.csv"
pd.DataFrame(new_data).to_csv(output_file_path, index=False)






