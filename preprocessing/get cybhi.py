#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:57:50 2023

@author: victorrincon
"""

import os

# Ruta donde se encuentran los archivos TXT
path = "/Users/victorrincon/Documents/Maestria/MISTI/Proyecto ECG/Codigos/data/raw/CYBHi/data/long-term"

# Lista de archivos TXT en la ruta especificada
txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]

# Diccionario para almacenar la cantidad de archivos por inicial
initial_counts = {}

# Iterar sobre la lista de archivos
for filename in txt_files:
    # Obtener la inicial del nombre de archivo
    initial = filename[0]
    # Si la inicial ya existe en el diccionario, aumentar el contador
    if initial in initial_counts:
        initial_counts[initial] += 1
    # Si no existe, agregarla al diccionario con un valor de 1
    else:
        initial_counts[initial] = 1
