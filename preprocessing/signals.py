# ECG Biometric Authentication using Transformer model

import os

from scipy.signal import filtfilt
import pandas as pd
import numpy as np
import wfdb
import math 

def filters(array, n):
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    array = filtfilt(b, a, array, padlen=50)
    return array


# Sampling rate, sr for MIT-BIH is 360 Hz
# Get data, convert .dat to .csv files
def constructor(directory, filename, db):
    
    if db == "fantasiadb":
        signals, fields = wfdb.rdsamp(os.path.join(directory, filename),250)
        a = [m[1] for m in signals]  # get filtered signals
        a = ([x for x in a if math.isnan(x) == False])
        df = pd.DataFrame(a)
        df.to_csv(os.path.join(directory, filename + '.' 'csv'), index=False)
        
    else:
        
    
        signals, fields = wfdb.rdsamp(os.path.join(directory, filename))
        # Read from online Physionet Dataset
        # signals, fields = wfdb.rdsamp(filename, pn_dir=db)
        a = [m[0] for m in signals]  # get filtered signals
    
        df = pd.DataFrame(a)
        df.to_csv(os.path.join(directory, filename + '.' 'csv'), index=False)


class GetSignals:
    def __init__(self):
        self.mit_dir = os.path.expanduser("data/raw/mit/")
        #self.bmd_dir = os.path.expanduser("data/raw/bmd101/")
        self.ecg_id = os.path.expanduser("data/raw/ecgid/")
        self.ptb_dir = os.path.expanduser("data/raw/ptb/")
        self.fantasia_dir = os.path.expanduser("data/raw/fantasia/")
        self.CYBHi_dir = os.path.expanduser("data/raw/CYBHi/data/long-term-csv")
        
        
        self.mitdb = 'mitdb'
        self.ecgiddb = 'ecgiddb'
        self.ptbdb = "ptbdb"
        self.fantasiadb = "fantasiadb"
        self.CYBHidb = "CYBHidb"

    def mit(self, people):
        # rastrea cada carpeta y envía el archivo .dat al constructor
        print('Converting to .dat to .csv...')
        files = sorted(os.listdir(self.mit_dir))
        print(len(files), " files found.\n")
        for file in files:
            if file.endswith('.dat') and file.replace(".dat", "") in people:
                basename = file.split('.')[0]
                constructor(self.mit_dir, basename, self.mitdb)
                print('Person ' + basename)
                
    def fantasia(self, people):
        # Rastrea cada carpeta y envía el archivo .dat al constructor 
        print("Convertir .dat a .csv... ")
        files = sorted(os.listdir(self.fantasia_dir))
        print(len(files), " archivos encontrados.\n")
        for file in files:
            if file.endswith(".dat") and file.replace(".dat", "") in people:
                basename = file.split(".")[0]
                constructor(self.fantasia_dir, basename, self.fantasiadb)
                print("Persona " + basename)
    
    def ptb(self, people):
        #rastrea cada carpeta y envía el archivo .dat al constructor
        print("Convertir .dat a .csv...")
        folders = sorted(os.listdir(self.ptb_dir))
        #print(len(folders), " Folders found.\n")
        # print(folders)
        
        count = 0
        for folder in folders:
            if folder.startswith("patient_") and folder.replace("patient_", "") in people:
                 records = sorted(os.listdir(os.path.join(self.ptb_dir, folder)))
                 print(len(records), " records found.\n")
                 for record in records:
                     basename = record.split(".",1)[0]
                     constructor(self.ptb_dir + folder, basename, self.ptbdb) 
                 count += 1
                 print("Patient " + str(count))

            
        # count = 0 
        # for folder in folders:
        #     #print(folder)
        #     if folder.endswith("patient_") and folder.replace("patient_", "") in people:
        #         records = sorted(os.listdir(os.path.join(self.ptb_dir, folder)))
        #         print(len(records), " records found.\n")
        #         for record in records:
        #             basename = record.split('.', 1)[0]
        #             constructor(self.ptb_dir + folder, basename, self.ptbdb)
        #         count += 1
        #         print('Person ' + str(count))
                    

    
    def bmd(self, people):
        folders = sorted(os.listdir(self.bmd_dir + "/raw/"))
        for folder in folders:
            if not folder.startswith('.') and folder in people:
                files = sorted(os.listdir(os.path.join(self.bmd_dir + "/raw/", folder)))
                print(len(files), " files found.\n")
                for file in files:
                    if file.startswith('ECGLog'):
                        name = self.bmd_dir + "/raw/" + folder + "/" + file
                        count = 0
                        array = []
                        with open(name, 'r') as f:
                            for line in f:
                                count += 1
                                if count == 1:
                                    continue
                                value = int(line.strip().split()[1])
                                array.append(value)

                        array = np.array(array, dtype="float32")
                        array = np.interp(array, (array.min(), array.max()), (-1, +1))
                        array = np.array(array, dtype="float32")
                        unfiltered = array[:]

                        df = pd.DataFrame()
                        df["0"] = array[:]
                        df.to_csv(self.bmd_dir + "csv/" + folder + '.' 'csv', index=False)
                        print("Person:", str(folder))

    def ecgid(self, people):

        # crawls into every folder and sends .dat file to constructor
        print('Converting to .dat to .csv...')
        folders = sorted(os.listdir(self.ecg_id))
        count = 0
        for folder in folders:
            if folder.startswith('Person_') and folder.replace("Person_", "") in people:
                records = sorted(os.listdir(os.path.join(self.ecg_id, folder)))
                print(len(records), " records found.\n")
                for record in records:
                    # only get the first 2 records for all people to have equal weights
                    if record.startswith('rec_') or record.endswith('.dat'):
                        basename = record.split('.', 1)[0]
                        constructor(self.ecg_id + folder, basename, self.ecgiddb)
                count += 1
                print('Person ' + str(count))
