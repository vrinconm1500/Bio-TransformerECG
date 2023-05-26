#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:51:00 2023

@author: victormelchor
"""


"""  Programa para leer las bases de datos de señales ECG, procesarlas y guardar las lecturas en 
    formato CSV 

"""


#Importamos librerias

from signals import GetSignals
from features import GetFeatures
from setup import Setup
import os
import pandas as pd 


# Base de datos ECGID 

# Generamos lista de sujetos

# exclude person 74
ecgid = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
         '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
         '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
         '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
         '71', '72', '73', '75', '76', '77', '78', '79', '80',
         '81', '82', '83', '84', '85', '86', '87', '88', '89', '90']

gs = GetSignals()
gs.ecgid(ecgid)

feats = GetFeatures()
feats.features('ecgid', ecgid)

su = Setup()
su.load_signals(290, "ecgid_250", ecgid[:90], 0)




### MIT BIH

mit = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
       '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
       '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
       '222', '223', '228', '230', '231', '232', '233', '234']

gs = GetSignals()
gs.mit(mit)

feats = GetFeatures()
feats.features('mit', mit)

su = Setup()
su.load_signals(290, "mit-bih-290", mit[:48], 0)








#Generamos la lista de todos los pacientes de la base de datos 

ruta="/Users/victorrincon/Documents/ECG-CODE/biometricECG-main/data/raw/ptb"
#nombres = []  #creamos una variable listado vacia para guardar los nombres
ptb = []   #creamos una variable ptb vacia para guardar el numero de los pacientes
for item in os.listdir(ruta): #si ponemos "." significa que hara el listado en el mismo directorio donde esta el fichero de python
    
    try:
        int(item[8:])
        #nombres.append(item) #guardamos cada nombre de archivo del directo destino en esta variable
        ptb.append(item[8:])  # guardamos el numero del archivo
    except:
        pass
	
ptb.sort()
# print("Listado de Todos los nombres", nombres)



gs = GetSignals()
gs.ptb(ptb)


features = pd.read_csv("/Users/victormelchor/Documents/biometricECG-main/data/raw/ptb/patient_001/s0010_re.csv")
filtered = features['0'].values



feats = GetFeatures()
feats.features('ptb', ptb)


su = Setup()
su.load_signals(290, "ptb_290", ptb[:290], 0)




# Creamos la base de datos de unicamente los pacientes sanos 

import pandas as pd 

ruta="/Users/victorrincon/Documents/ECG-CODE/biometricECG-main/data/raw/ptb"
#nombres = []  #creamos una variable listado vacia para guardar los nombres
ptb = []   #creamos una variable listado vacia para guardar el numero de las fotos
for item in os.listdir(ruta): #si ponemos "." significa que hara el listado en el mismo directorio donde esta el fichero de python
    
    try:
        int(item[8:])
        #nombres.append(item) #guardamos cada nombre de archivo del directo destino en esta variable
        ptb.append(item[8:])  # guardamos el numero del archivo
    except:
        pass
sanos = []
for person in ptb:
    folder = os.path.expanduser("data/raw/ptb/patient_" + person + "/")
    files = []
    for item in os.listdir(folder):
        if item[-4:] == ".hea":
            files.append(folder + item)
            print(files)
    for file in files:
        with open(file, 'r') as f:
            diagnostic = pd.read_fwf(f)
            strr = diagnostic.iloc[19, 0]
            if strr.find("Healthy") != -1:
                sanos.append((person))
    
from collections import Counter

sanos = list(Counter(sanos).keys())
sanos.sort()

gs = GetSignals()
gs.ptb(sanos)



feats = GetFeatures()
feats.features('ptb', sanos)


su = Setup()
su.load_signals(290, "ptb-sanos_290", sanos[:290], 0)



##### Base de datos fantasia 

ruta="/Users/victorrincon/Documents/ECG-CODE/biometricECG-main/data/raw/fantasia"
#nombres = []  #creamos una variable listado vacia para guardar los nombres
fantasia = []   #creamos una variable ptb vacia para guardar el numero de los pacientes
for item in os.listdir(ruta): #si ponemos "." significa que hara el listado en el mismo directorio donde esta el fichero de python
    if item.endswith(".dat"):
        fantasia.append(item[:5])
fantasia.sort()
#fantasia = ["f1o01"]

gs = GetSignals()
gs.fantasia(fantasia)


feats = GetFeatures()

feats.features('fantasia', fantasia)

su = Setup()
su.load_signals(290, "fantasia_290", fantasia[:40], 0)


###### CYBHi long term 


CYBHi = ['RA', 'MP', 'ARA', 'MJR', 'JM', 'MGA', 'MA', 'JP', 'MB', 'AC', 'CF', 
         'FP', 'IC', 'ABD', 'SF', 'JV', 'CB', 'FP', 'JM', 'PMA', 'JPA', 'SR', 
         'JA', 'FM', 'ARA', 'AG', 'VO', 'JN', 'IB', 'MGA', 'MJR', 'DS', 'FO', 
         'RD', 'RAA', 'TC', 'JS', 'TV', 'JPA', 'JCA', 'JA', 'CB', 'AL', 'AC', 
         'CF', 'RL', 'MMJ', 'JL', 'PMA', 'JCA', 'IC', 'AA', 'CSR', 'AG', 'RF', 
         'HF', 'JB', 'ARL', 'ABD', 'AR', 'IB', 'MB', 'RF', 'ACA', 'RA', 'MP', 
         'JP', 'JSA', 'DB', 'SR', 'RAA', 'DC', 'JCC', 'FM', 'TF', 'PM', 'PES', 
         'VO', 'ARF', 'MQ', 'RRA', 'SF', 'GF', 'MC', 'ACA', 'JSA', 'MQ', 'VM', 
         'JL', 'TC', 'PES', 'MMJ', 'JC', 'AFS', 'TF', 'JCC', 'TV', 'ARL', 'JS', 
         'JB', 'MA', 'GF', 'VM', 'FO',   'AR',  'DC', 'CSR', 'RR', 'DS', 'MBA', 
         'AL', 'RD', 'AFS', 'MBA', 'JV', 'RR', 'MC', 'JC', 'RRA', 'HF', 'AA', 
         'ARF', 'RL', 'DB', 'PM', 'JN']




feats = GetFeatures()
feats.features('CYBHi', CYBHi)


su = Setup()
su.load_signals(290, "CYBHi-290", CYBHi[:126], 0)


