#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:08:31 2023

@author: victormelchor
"""

import os

nombres = []  #creamos una variable listado vacia para guardar los nombres
numero = []   #creamos una variable listado vacia para guardar el numero de las fotos
for item in os.listdir("."): #si ponemos "." significa que hara el listado en el mismo directorio donde esta el fichero de python
	# si queremos poner una direccion como por ejemplo C:\\Fotos\\RivieraMaya ponemos simpre doble barra para que python no se confunda
    # if item[-4:] == ".JPG":  #Si queremos filtro solo los archivos JPG
    
    nombres.append(item) #guardamos cada nombre de archivo del directo destino en esta variable
    numero.append(item[9:])  # guardamos el numero del archivo

item = item[9:]
div = item.split("-",maxsplit=1)



#Cambiar nombres a la base de datos CYBHi


nombres = []  #creamos una variable listado vacia para guardar los nombres
numero = []   #creamos una variable listado vacia para guardar el numero de las fotos
for item in os.listdir("."): #si ponemos "." significa que hara el listado en el mismo directorio donde esta el fichero de python
	# si queremos poner una direccion como por ejemplo C:\\Fotos\\RivieraMaya ponemos simpre doble barra para que python no se confunda
    # if item[-4:] == ".JPG":  #Si queremos filtro solo los archivos JPG
    
    nombres.append(item) #guardamos cada nombre de archivo del directo destino en esta variable
    item = item[9:]
    div = item.split("-",maxsplit=1)
    numero.append(div[0])  # guardamos el numero del archivo


import numpy as np
unique, counts = np.unique(numero, return_counts=True)

result = np.column_stack((unique, counts)) 
print (result)


# print("Listado de Todos los nombres", nombres)

numerodestino = [] # guardamos el numero de destino
cambio = []		   # guardamos el nombre de destino




for index,i in enumerate(numero):  #index es el numero de registro y i es el numero del rango 1014 a 1502
    numerodestino.append(i) # guardamos en un array todos los numeros de destino
    


    cambio.append("patient_" + str(numerodestino[index]))  #creamos el nombre del archivo destino con DSC_ convertimos a string el numero y añadimos la extension
    os.rename(nombres[index], cambio[index])  #aqui ponemos los nombre originales y los nombres destino,
    
    print(nombres[index] ) #mostramos por pantalla los nombre originales
    print(cambio[index] )  #mostramos por pantalla los nombres de destino
    print(nombres[index][-4:] ) #nos muestra la extension del archivo original

    print("Nombre Cambiado",cambio)