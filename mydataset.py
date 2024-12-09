from torch.utils.data import Dataset
import pandas as pd
import torchaudio.functional as F
import torchaudio
import glob
import torch
class MyDataset(Dataset):
    def __init__(self, dataloc, csv, transform):
        self.datafiles = glob.glob(dataloc)
        self.labels = pd.read_csv(csv)
        self.labels = {row['itemid']: row['hasbird'] for _, row in self.labels.iterrows()}
        self.transform = transform

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, i):
        file = self.datafiles[i]
        mel_spec = self.transform(file) 
        file_id = file.split('/')[-1].split('.')[0]
        label = self.labels[file_id]
        
        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.float32) 

        return mel_spec, label, file_id
