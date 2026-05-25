import os
import numpy as np
import math
import scipy

# Default data location. Override with ARRAY_SAFETY_DATA.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ARRAY_SAFETY_DATA",
    os.path.normpath(os.path.join(_HERE, "..", "data", "Data")),
)


def _require_data_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "See release/data/README.md for instructions on the shipped and "
            "external waveform/array files."
        )
    return path


class Waveform():
    def __init__(self):
        iq_path = _require_data_file("LFM_1280MHz_IBW.mat")
        IQ = scipy.io.loadmat(iq_path)
        self.IQ = IQ['IQ'] 
        
        self.fs = 1.6 # GHz
        self.fc = 3 # GHz
        self.freq_signal = np.linspace(-self.fs/2, self.fs/2, self.IQ.shape[0]) + self.fc  # Frequency range for IQ data
        self.num_samples  = len(self.IQ)
        self.bw = None
    
    def generate_waveform(self,bw):
        if bw > (self.fs * 1e9)/2:
            raise ValueError("Bandwidth cannot be greater than half of sampling frequency.")
        
        self.bw = bw
        stop_time = 1/self.fs * 1e-9 * self.num_samples
        time = np.linspace(0,stop_time,self.num_samples)
         # The chirp rate (how fast the frequency changes)
        chirp_rate = bw / stop_time
        
        # Calculate the instantaneous phase of the LFM signal
        # The formula for the phase of a baseband LFM chirp is:
        # phi(t) = 2 * pi * ((-bandwidth / 2) * t + (chirp_rate / 2) * t^2)
        phase = 2 * np.pi * ((-bw / 2) * time + (chirp_rate)/2  * time**2)
        
        # Generate the complex IQ samples using Euler's formula: exp(j*phi) = cos(phi) + j*sin(phi)
        self.IQ = np.exp(1j * phase)
        freq_signal = self.freq_signal * 1e9
        self.freq_signal_idx = (freq_signal >= -bw/2 + self.fc*1e9) & (freq_signal <= bw/2 + self.fc*1e9)

        if np.any(self.freq_signal_idx):
            self.freq_signal_idx[16000] = True
            
        self.freq_signal = self.freq_signal[self.freq_signal_idx]
        
        return self.IQ
    
    def get_fft_signal(self): 
        IQ_fft = np.fft.fftshift(np.fft.fft(self.IQ, axis=0,norm='forward'), axes=0)  # FFT along the time axis
        if not self.bw is None:
            IQ_fft = IQ_fft[self.freq_signal_idx]
        return IQ_fft

        
