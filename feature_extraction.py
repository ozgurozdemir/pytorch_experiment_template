import time
import numpy as np

import librosa

class FeatureExtraction():
    def __init__(self, sr, **kwargs):
        self.SR = sr
        self.kwargs = kwargs
    
    # Mean length of waves: 64731.73
    def _trimming_padding(self, wave):
        duration = self.kwargs["trimming_padding_duration"]
        
        if wave.shape[0] < duration:
            pad  = np.zeros((duration - wave.shape[0],))
            wave = np.concatenate((wave, pad), axis=0)
        else:
            wave = wave[:duration]
        return wave
    
    def _calc_adaptive_window_length(self, wave):
        """
            Calculating adaptive window length considering the given time resolution
            for fixing the dimensions of the spectrograms, because the duration of 
            speeches varies. Solves the following formula :
            
            $n_{time} = \big(\frac{sample\_length}{window\_size} * 2\big) + 1$ 
        """
        sample_len = len(wave)
        nfft = int((2 * sample_len) / (self.kwargs["adaptive_time_resolution"] - 1)) 
        
        return nfft
    
  
    def _get_log_mel_spectrogram(self, wave):
        if "trimming_padding_duration" in self.kwargs:
            wave = self._trimming_padding(wave)
            nfft = self.kwargs["nfft"]
            
        if "adaptive_time_resolution" in self.kwargs:
            nfft = self._calc_adaptive_window_length(wave)
            
        wave = librosa.feature.melspectrogram(wave,  sr=self.SR, n_fft=nfft, hop_length=nfft // 2)
        wave = librosa.power_to_db(wave, ref=np.max)
        # print(wave.shape)

        # assuring time resolution, some lengths cannot be fixed due to int conversion 
        if "adaptive_time_resolution" in self.kwargs:
            return wave[:, :self.kwargs["adaptive_time_resolution"]] 
        else:
            return wave
        
        
    def prepare_log_mel_spectrogram(self, dataset):
        print(">> Preparing log-mel spectrograms...")
        start_time = time.time()
        feature_ds = {}
        
        for i, set_name in enumerate(dataset):
            waves  = [self._get_log_mel_spectrogram(sample["wave"]) for sample in dataset[set_name]]
            labels = [sample["label"] for sample in dataset[set_name]]
            
            feature_ds[i] = {
                "wave":  np.array(waves),
                "label": np.array(labels)
            }
        
        print(f":: Features are extracted in {time.time()-start_time} sec...")
        return feature_ds