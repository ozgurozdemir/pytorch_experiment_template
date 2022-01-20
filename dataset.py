# Date: 18.01.2022

import os, time
import numpy as np
import pandas as pd
import librosa

import torch


class SpeechEmotionDataset():
    """
        attr
            dataset:    dataset after feature extraction. empty if it will be read from numpy
            batch_size: mini-batch size used for training 
    """
    def __init__(self, dataset, batch_size):
        self.dataset    = dataset
        self.batch_size = batch_size    
    
    
    def read_from_numpy(self, file_path):
        print(">> Reading feature dataset from numpy...")
        self.dataset = np.load(file_path, allow_pickle=True)
        self.dataset = self.dataset["dataset"][0]
    
    
    def prepare_speaker_independent_sets(self, train_set, valid_set, test_set):
        print(">> Preparing speaker independent sets...")
        
        # concatenating the individiual speakers
        X_train = np.concatenate([self.dataset[t]["wave"]  for t in train_set]) 
        y_train = np.concatenate([self.dataset[t]["label"] for t in train_set])
        
        if len(valid_set) > 0:
            X_valid = np.concatenate([self.dataset[v]["wave"]  for v in valid_set])
            y_valid = np.concatenate([self.dataset[v]["label"] for v in valid_set]) 

        X_test = np.concatenate([self.dataset[t]["wave"]  for t in test_set])
        y_test = np.concatenate([self.dataset[t]["label"] for t in test_set])
        
        # informing about the distribution of the emotions
        self._print_info("train", y_train)
        if len(valid_set) > 0: 
            self._print_info("valid", y_valid)
        self._print_info("test",  y_test)
        
        if len(valid_set) > 0 :
            return (self._prepare_torch_dataset(X_train, y_train),
                    self._prepare_torch_dataset(X_valid, y_valid),
                    self._prepare_torch_dataset(X_test,  y_test))
        else:
            return (self._prepare_torch_dataset(X_train, y_train),
                    self._prepare_torch_dataset(X_test,  y_test))
    
    
    # TODO: prepare generic dataset for speaker dependent configuration
    def prepare_speaker_dependent_sets(self, valid_size, test_size):
        pass
        
    
    def _prepare_torch_dataset(self, X, y):
        X = np.expand_dims(X, axis=1) # creating channel dim
        X = torch.Tensor(X) 
        y = torch.Tensor(y)
        
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
    
    
    def _print_info(self, set_info, labels):
        print(f":: Distribution for {set_info} ".ljust(26) + 
              f"- [neu, sad, ang, hap]: " + str(labels.sum(axis=0)).ljust(24) + 
              f"-> " + str(labels.sum()))


class IEMOCAP:
    def __init__(self, path, sr):
        self.path = path
        self.SR   = sr
        
        self.emotion_labels = ["neu", "sad", "ang", "hap"]
        self.emotion_vector = {
            "neu": [1, 0, 0, 0],
            "sad": [0, 1, 0, 0],
            "ang": [0, 0, 1, 0],
            "hap": [0, 0, 0, 1] 
        }
            
    def read_from_numpy(self, file_name):
        print(">> Reading dataset from numpy file")
        start_time = time.time()
        
        dataset = np.load(f"{self.path}/{file_name}", allow_pickle=True)
        dataset = dataset["dataset"][0]
        
        print(f":: Dataset is read in {time.time()-start_time} sec...")
        return dataset
    
    
    def read_from_folder(self):
        dataset = {
            "Session1": [],
            "Session2": [],
            "Session3": [],
            "Session4": [],
            "Session5": []
        }
        
        print(">> Reading dataset from folder")
        start_time = time.time()
        
        for i in range(1, 6):
            dataset[f"Session{i}"] = self._prepare_session_dict(i)
            
        print(f":: Dataset is read in {time.time()-start_time} sec...")
        return [dataset]
        
        
    def _prepare_session_dict(self, ses):
        session = []

        # reading metadata
        metadata    = pd.read_csv(f"{self.path}/Ses0{ses}_desc.csv")
        transcripts = pd.read_csv(f"{self.path}/Ses0{ses}_trans.csv", sep='\n', header=None)

        # format the sample
        for i, meta in enumerate(metadata.iloc):
            
            if meta["Emotion"] in self.emotion_labels:
                data = {}

                data["wave"], _    = librosa.load(f"{path}/wav2/{meta['SentID']}.wav", sr=self.SR)
                data["label"]      = self._convert_label_vector(meta["Emotion"])
                data["speaker"]    = meta["Speaker"]
                data["type"]       = "impro" if "impro" in meta["SessionID"] else "script"
                data["vad"]        = [meta["Valence"], meta["Arousal"], meta["Dominance"]]
                data["transcript"] = transcripts.values[i][0]

                session.append(data)

        return session

    
    def _convert_label_vector(self, label):
        return self.emotion_vector[label]
    
# TODO: Prepare RAVDESS dataset class