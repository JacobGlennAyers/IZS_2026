import torchaudio
import librosa
import numpy as np
import torch

'''
def pad_waveform(waveform, target_length=510):
    """
    Guaranteed safe padding function that works for any length input.
    
    Args:
        waveform (torch.Tensor or array-like): Input audio waveform
        target_length (int): Desired output length (default: 512)
        
    Returns:
        torch.Tensor: Padded waveform with exactly the target length
    """
    import torch
    import numpy as np
    
    # Convert to tensor if needed
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Store original shape and device
    device = waveform.device
    original_shape = waveform.shape
    
    # Handle the case where waveform is shape [1, length] (channel first format)
    if len(waveform.shape) == 2 and waveform.shape[0] == 1:
        current_length = waveform.shape[1]
        
        # Already the right length
        if current_length == target_length:
            return waveform
            
        # Create a new tensor of the right size
        result = torch.zeros((1, target_length), dtype=waveform.dtype, device=device)
        
        # Fill with repetitions of the original
        if current_length > 0:  # Only if we have data
            # Calculate how many full repetitions we need
            repetitions = (target_length + current_length - 1) // current_length
            
            # Create repeated version
            repeated = waveform.repeat(1, repetitions)
            
            # Take exactly what we need
            result[0, :] = repeated[0, :target_length]
            
        return result
        
    # Handle 1D tensor case
    elif len(waveform.shape) == 1:
        current_length = waveform.shape[0]
        
        # Already the right length
        if current_length == target_length:
            return waveform
            
        # Create a new tensor of the right size
        result = torch.zeros(target_length, dtype=waveform.dtype, device=device)
        
        # Fill with repetitions of the original
        if current_length > 0:  # Only if we have data
            # Calculate how many full repetitions we need
            repetitions = (target_length + current_length - 1) // current_length
            
            # Create repeated version
            repeated = waveform.repeat(repetitions)
            
            # Take exactly what we need
            result[:] = repeated[:target_length]
            
        return result
    
#def load_audio(path, offset, duration, mono=True):
#    pass



class AudioLoader(torch.nn.Module):
    """
    Callable audio loader that can be used similarly to transform functions
    """
    def __init__(self, experiment_parameters):
        super().__init__()
        self.audio_normalization = experiment_parameters["audio_normalization"]
        self.target_sample_rate = experiment_parameters["sample_rate"]
        self.target_length = experiment_parameters["spectrogram_parameters"]["n_fft"]
        
        # Pre-create resampling transform if needed
        self.resample_transform = None
        if self.target_sample_rate is not None:
            # Will be updated with actual source sample rate during forward pass
            pass
    
    def forward(self, path, offset, duration, mono=True):
        """
        Load and process audio file
        """
        # Loading Signal
        if self.audio_normalization is None:
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
        elif self.audio_normalization == "min_max":
            SIGNAL, SAMPLE_RATE = load_audio_min_max(path, offset, duration, mono=mono)
        elif self.audio_normalization == "z_score":
            SIGNAL, SAMPLE_RATE = load_audio_z_score(path, offset, duration, mono=mono)
        
        # Resampling Signal
        if self.target_sample_rate is not None:
            # Create or update resampling transform
            if self.resample_transform is None or self.resample_transform.orig_freq != SAMPLE_RATE:
                self.resample_transform = torchaudio.transforms.Resample(
                    orig_freq=SAMPLE_RATE, 
                    new_freq=self.target_sample_rate
                )
            SIGNAL = self.resample_transform(torch.tensor(SIGNAL).unsqueeze(0))
            SAMPLE_RATE = self.target_sample_rate
        
        SIGNAL = pad_waveform(SIGNAL, target_length=self.target_length)
        return SIGNAL, SAMPLE_RATE
    
    def __call__(self, path, offset, duration, mono=True):
        """
        Make the class callable like other transform functions
        """
        return self.forward(path, offset, duration, mono)

def load_audio(experiment_parameters):
    """
    Function that returns a callable audio loader, matching the pattern of feature_vector_function
    """
    return AudioLoader(experiment_parameters)

def load_audio_z_score(path, offset, duration, mono=True):
    # Loading Signal
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    # Normalizing Signal
    SIGNAL = (SIGNAL - np.mean(SIGNAL)) / np.std(SIGNAL)

    return SIGNAL, SAMPLE_RATE
def load_audio_min_max(path, offset, duration, mono=True):
    # Loading Signal
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    # Normalizing Signal
    SIGNAL = (SIGNAL - np.min(SIGNAL)) / (np.max(SIGNAL) - np.min(SIGNAL))
    # For -1 to 1 range
    SIGNAL = 2 * SIGNAL - 1
    #SIGNAL = (SIGNAL - np.mean(SIGNAL))/np.std(SIGNAL)

    return SIGNAL, SAMPLE_RATE

class SigmoidTransform(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

def feature_vector_function(experiment_parameters):
    """
    Function that encapsulates different potential spectrogram feature vectors
    """
    params = experiment_parameters["spectrogram_parameters"].copy()
    
    if experiment_parameters["spectrogram_type"] == "Mel":
        # MelSpectrogram requires sample_rate
        fn = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(**params),
            SigmoidTransform()
        )
    elif experiment_parameters["spectrogram_type"] == "STFT":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            SigmoidTransform()
        )
    elif experiment_parameters["spectrogram_type"] == "STFT_dB":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            torchaudio.transforms.AmplitudeToDB(),
            SigmoidTransform()  # Apply sigmoid to map to [0,1]
        )
        
    
    return fn

class EnhancedAudioProcessor:
    """
    Wrapper around your existing feature_vector_function that adds support for pre-loaded audio
    """
    def __init__(self, experiment_parameters):
        self.experiment_parameters = experiment_parameters
        self.audio_loader = load_audio(experiment_parameters)  # Your existing audio loader
        self.spectrogram_fn = feature_vector_function(experiment_parameters)  # Your existing spectrogram function
        
    def __call__(self, filepath, onset=0.0, duration=None):
        """Handle file-based audio loading (your existing path)"""
        # Load and normalize audio
        waveform, sample_rate = self.audio_loader(filepath, onset, duration)
        # Convert to spectrogram
        spectrogram = self.spectrogram_fn(waveform)
        # Normalize spectrogram
        spectrogram = spectrogram / torch.max(spectrogram)
        return spectrogram
    
    def process_preloaded_audio(self, waveform, sample_rate):
        """Handle pre-loaded audio (new functionality for YESNO/TIMIT)"""
        # Ensure waveform is in the right format
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
        # Apply normalization
        if self.experiment_parameters["audio_normalization"] == "z_score":
            mean = torch.mean(waveform)
            std = torch.std(waveform)
            waveform = (waveform - mean) / std
        elif self.experiment_parameters["audio_normalization"] == "min_max":
            min_val = torch.min(waveform)
            max_val = torch.max(waveform)
            waveform = (waveform - min_val) / (max_val - min_val)
            waveform = 2 * waveform - 1  # Scale to [-1, 1]
        
        # Resample if needed
        if sample_rate != self.experiment_parameters["sample_rate"]:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.experiment_parameters["sample_rate"]
            )
            waveform = resample_transform(waveform)
        
        # Pad waveform
        waveform = pad_waveform(waveform, target_length=self.experiment_parameters["spectrogram_parameters"]["n_fft"])
        
        # Convert to spectrogram
        spectrogram = self.spectrogram_fn(waveform)
        
        # Normalize spectrogram
        spectrogram = spectrogram / torch.max(spectrogram)
        
        return spectrogram

        '''


import torchaudio
import librosa
import numpy as np
import torch

def pad_waveform(waveform, target_length=510):
    """
    Guaranteed safe padding function that works for any length input.
    
    Args:
        waveform (torch.Tensor or array-like): Input audio waveform
        target_length (int): Desired output length (default: 512)
        
    Returns:
        torch.Tensor: Padded waveform with exactly the target length
    """
    import torch
    import numpy as np
    
    # Convert to tensor if needed
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Store original shape and device
    device = waveform.device
    original_shape = waveform.shape
    
    # Handle the case where waveform is shape [1, length] (channel first format)
    if len(waveform.shape) == 2 and waveform.shape[0] == 1:
        current_length = waveform.shape[1]
        
        # Already the right length
        if current_length == target_length:
            return waveform
            
        # Create a new tensor of the right size
        result = torch.zeros((1, target_length), dtype=waveform.dtype, device=device)
        
        # Fill with repetitions of the original
        if current_length > 0:  # Only if we have data
            # Calculate how many full repetitions we need
            repetitions = (target_length + current_length - 1) // current_length
            
            # Create repeated version
            repeated = waveform.repeat(1, repetitions)
            
            # Take exactly what we need
            result[0, :] = repeated[0, :target_length]
            
        return result
        
    # Handle 1D tensor case
    elif len(waveform.shape) == 1:
        current_length = waveform.shape[0]
        
        # Already the right length
        if current_length == target_length:
            return waveform
            
        # Create a new tensor of the right size
        result = torch.zeros(target_length, dtype=waveform.dtype, device=device)
        
        # Fill with repetitions of the original
        if current_length > 0:  # Only if we have data
            # Calculate how many full repetitions we need
            repetitions = (target_length + current_length - 1) // current_length
            
            # Create repeated version
            repeated = waveform.repeat(repetitions)
            
            # Take exactly what we need
            result[:] = repeated[:target_length]
            
        return result


class AudioLoader(torch.nn.Module):
    """
    Callable audio loader that can be used similarly to transform functions
    """
    def __init__(self, experiment_parameters):
        super().__init__()
        self.audio_normalization = experiment_parameters["audio_normalization"]
        self.target_sample_rate = experiment_parameters["sample_rate"]
        self.target_length = experiment_parameters["spectrogram_parameters"]["n_fft"]
        
        # Pre-create resampling transform if needed
        self.resample_transform = None
        if self.target_sample_rate is not None:
            # Will be updated with actual source sample rate during forward pass
            pass
    
    def forward(self, path, offset, duration, mono=True):
        """
        Load and process audio file
        """
        # Loading Signal
        if self.audio_normalization is None:
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
        elif self.audio_normalization == "min_max":
            SIGNAL, SAMPLE_RATE = load_audio_min_max(path, offset, duration, mono=mono)
        elif self.audio_normalization == "z_score":
            SIGNAL, SAMPLE_RATE = load_audio_z_score(path, offset, duration, mono=mono)
        
        # Resampling Signal
        if self.target_sample_rate is not None:
            # Create or update resampling transform
            if self.resample_transform is None or self.resample_transform.orig_freq != SAMPLE_RATE:
                self.resample_transform = torchaudio.transforms.Resample(
                    orig_freq=SAMPLE_RATE, 
                    new_freq=self.target_sample_rate
                )
            SIGNAL = self.resample_transform(torch.tensor(SIGNAL).unsqueeze(0))
            SAMPLE_RATE = self.target_sample_rate
        
        SIGNAL = pad_waveform(SIGNAL, target_length=self.target_length)
        return SIGNAL, SAMPLE_RATE
    
    def __call__(self, path, offset, duration, mono=True):
        """
        Make the class callable like other transform functions
        """
        return self.forward(path, offset, duration, mono)

def load_audio(experiment_parameters):
    """
    Function that returns a callable audio loader, matching the pattern of feature_vector_function
    """
    return AudioLoader(experiment_parameters)

def load_audio_z_score(path, offset, duration, mono=True):
    """Load audio and normalize using z-score (standardization)"""
    # Loading Signal
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    # Normalizing Signal
    SIGNAL = (SIGNAL - np.mean(SIGNAL)) / np.std(SIGNAL)
    return SIGNAL, SAMPLE_RATE


def load_audio_min_max(path, offset, duration, mono=True):
    """Load audio and normalize using min-max scaling to [-1, 1]"""
    # Loading Signal
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    # Normalizing Signal
    SIGNAL = (SIGNAL - np.min(SIGNAL)) / (np.max(SIGNAL) - np.min(SIGNAL))
    # For -1 to 1 range
    SIGNAL = 2 * SIGNAL - 1
    return SIGNAL, SAMPLE_RATE


def load_audio_mean_amplitude(path, offset, duration, mono=True):
    """Load audio and normalize by mean absolute amplitude"""
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    mean_amp = np.mean(np.abs(SIGNAL))
    if mean_amp > 0:
        SIGNAL = SIGNAL / mean_amp
    SIGNAL = np.clip(SIGNAL, -10.0, 10.0)
    return SIGNAL, SAMPLE_RATE


def load_audio_mean_intensity(path, offset, duration, mono=True):
    """Load audio and normalize by RMS (Root Mean Square) intensity"""
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
    rms = np.sqrt(np.mean(SIGNAL ** 2))
    if rms > 0:
        SIGNAL = SIGNAL / rms
    SIGNAL = np.clip(SIGNAL, -10.0, 10.0)
    return SIGNAL, SAMPLE_RATE


class AudioLoader(torch.nn.Module):
    """
    Callable audio loader that can be used similarly to transform functions
    """
    def __init__(self, experiment_parameters):
        super().__init__()
        self.audio_normalization = experiment_parameters["audio_normalization"]
        self.target_sample_rate = experiment_parameters["sample_rate"]
        self.target_length = experiment_parameters["spectrogram_parameters"]["n_fft"]
        
        # Pre-create resampling transform if needed
        self.resample_transform = None
        if self.target_sample_rate is not None:
            # Will be updated with actual source sample rate during forward pass
            pass
    
    def forward(self, path, offset, duration, mono=True):
        """
        Load and process audio file
        """
        # Loading Signal
        if self.audio_normalization is None:
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
        elif self.audio_normalization == "min_max":
            SIGNAL, SAMPLE_RATE = load_audio_min_max(path, offset, duration, mono=mono)
        elif self.audio_normalization == "z_score":
            SIGNAL, SAMPLE_RATE = load_audio_z_score(path, offset, duration, mono=mono)
        elif self.audio_normalization in ["mean_amplitude", "mean-amplitude"]:
            SIGNAL, SAMPLE_RATE = load_audio_mean_amplitude(path, offset, duration, mono=mono)
        elif self.audio_normalization in ["mean_intensity", "mean-intensity"]:
            SIGNAL, SAMPLE_RATE = load_audio_mean_intensity(path, offset, duration, mono=mono)
        
        # Resampling Signal
        if self.target_sample_rate is not None:
            # Create or update resampling transform
            if self.resample_transform is None or self.resample_transform.orig_freq != SAMPLE_RATE:
                self.resample_transform = torchaudio.transforms.Resample(
                    orig_freq=SAMPLE_RATE, 
                    new_freq=self.target_sample_rate
                )
            SIGNAL = self.resample_transform(torch.tensor(SIGNAL).unsqueeze(0))
            SAMPLE_RATE = self.target_sample_rate
        
        SIGNAL = pad_waveform(SIGNAL, target_length=self.target_length)
        return SIGNAL, SAMPLE_RATE
    
    def __call__(self, path, offset, duration, mono=True):
        """
        Make the class callable like other transform functions
        """
        return self.forward(path, offset, duration, mono)


def load_audio(experiment_parameters):
    """
    Function that returns a callable audio loader, matching the pattern of feature_vector_function
    """
    return AudioLoader(experiment_parameters)


class SigmoidTransform(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class LogTen(torch.nn.Module):
    def forward(self,x):
        return torch.log10(x+1e-8)
    
class Loge(torch.nn.Module):
    def forward(self,x):
        return torch.log(x+1e-8)


def feature_vector_function(experiment_parameters):
    """
    Function that encapsulates different potential spectrogram feature vectors
    """
    params = experiment_parameters["spectrogram_parameters"].copy()
    
    if experiment_parameters["spectrogram_type"] == "Mel":
        # MelSpectrogram requires sample_rate
        fn = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(**params),
            SigmoidTransform()
        )
    elif experiment_parameters["spectrogram_type"] == "STFT":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            SigmoidTransform()
        )
    elif experiment_parameters["spectrogram_type"] == "STFT_dB":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            torchaudio.transforms.AmplitudeToDB(),
            SigmoidTransform()  # Apply sigmoid to map to [0,1]
        )
    elif experiment_parameters["spectrogram_type"] == "STFT_ln":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            Loge(),
            SigmoidTransform()  # Apply sigmoid to map to [0,1]
        )
    elif experiment_parameters["spectrogram_type"] == "STFT_log10":
        # Remove sample_rate parameter if it exists as Spectrogram doesn't need it
        if "sample_rate" in params:
            params.pop("sample_rate")
        fn = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(**params),
            LogTen(),
            SigmoidTransform()  # Apply sigmoid to map to [0,1]
        )
        
    return fn


class EnhancedAudioProcessor:
    """
    Unified audio processor that integrates your existing functions
    and handles both file-based and pre-loaded audio
    """
    def __init__(self, experiment_parameters):
        self.experiment_parameters = experiment_parameters
        self.spectrogram_fn = feature_vector_function(experiment_parameters)  # Your existing spectrogram function
        
    def __call__(self, filepath, onset=0.0, duration=None):
        """Handle file-based audio loading (your existing vocallbase path)"""
        # Use your existing audio normalization functions directly
        if self.experiment_parameters["audio_normalization"] is None:
            waveform_np, sample_rate = librosa.load(filepath, offset=onset, duration=duration, sr=None, mono=True)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        elif self.experiment_parameters["audio_normalization"] in ["min_max", "min-max"]:
            waveform_np, sample_rate = load_audio_min_max(filepath, onset, duration)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        elif self.experiment_parameters["audio_normalization"] in ["z_score", "z-score"]:
            waveform_np, sample_rate = load_audio_z_score(filepath, onset, duration)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        elif self.experiment_parameters["audio_normalization"] in ["mean_amplitude", "mean-amplitude"]:
            waveform_np, sample_rate = load_audio_mean_amplitude(filepath, onset, duration)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        elif self.experiment_parameters["audio_normalization"] in ["mean_intensity", "mean-intensity"]:
            waveform_np, sample_rate = load_audio_mean_intensity(filepath, onset, duration)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        
        # Resample if needed
        if sample_rate != self.experiment_parameters["sample_rate"]:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.experiment_parameters["sample_rate"]
            )
            waveform = resample_transform(waveform)
        
        # Pad waveform using your existing function
        #print(waveform.shape)
        #waveform = pad_waveform(waveform, target_length=self.experiment_parameters["spectrogram_parameters"]["n_fft"])
        
        # Convert to spectrogram using your existing function
        spectrogram = self.spectrogram_fn(waveform)
        
        # Normalize spectrogram (your existing approach)
        #spectrogram = spectrogram / torch.max(spectrogram)
        
        return spectrogram
    
    def process_preloaded_audio(self, waveform, sample_rate):
        """Handle pre-loaded audio (for YESNO/TIMIT datasets)"""
        # Ensure waveform is in the right format
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
        # Apply normalization using the same logic as your existing functions
        if self.experiment_parameters["audio_normalization"] in ["z_score", "z-score"]:
            # Same logic as load_audio_z_score but for tensors
            mean = torch.mean(waveform)
            std = torch.std(waveform)
            waveform = (waveform - mean) / std
        elif self.experiment_parameters["audio_normalization"] in ["min_max", "min-max"]:
            # Same logic as load_audio_min_max but for tensors
            min_val = torch.min(waveform)
            max_val = torch.max(waveform)
            waveform = (waveform - min_val) / (max_val - min_val)
            waveform = 2 * waveform - 1  # Scale to [-1, 1]
        elif self.experiment_parameters["audio_normalization"] in ["mean_amplitude", "mean-amplitude"]:
            # Normalize by mean absolute amplitude
            mean_amp = torch.mean(torch.abs(waveform))
            if mean_amp > 0:  # Avoid division by zero
                waveform = waveform / mean_amp
            # Optional: clip to reasonable range to prevent extreme values
            waveform = torch.clamp(waveform, -10.0, 10.0)
        elif self.experiment_parameters["audio_normalization"] in ["mean_intensity", "mean-intensity"]:
            # Normalize by RMS (Root Mean Square) - represents intensity/power
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:  # Avoid division by zero
                waveform = waveform / rms
            # Optional: clip to reasonable range to prevent extreme values
            waveform = torch.clamp(waveform, -10.0, 10.0)
        
        # Resample if needed
        if sample_rate != self.experiment_parameters["sample_rate"]:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.experiment_parameters["sample_rate"]
            )
            waveform = resample_transform(waveform)
        
        # Pad waveform using your existing function
        #waveform = pad_waveform(waveform, target_length=self.experiment_parameters["spectrogram_parameters"]["n_fft"])
        
        # Convert to spectrogram using your existing function
        spectrogram = self.spectrogram_fn(waveform)
        
        # Normalize spectrogram (your existing approach)
        #spectrogram = spectrogram / torch.max(spectrogram)
        
        return spectrogram


# Factory function to create the audio processor (maintains your existing pattern)
def create_audio_processor(experiment_parameters):
    """
    Create an enhanced audio processor that works with your existing setup
    This replaces the spectrogram_fn in your dataset constructor
    """
    return EnhancedAudioProcessor(experiment_parameters)