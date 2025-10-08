import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import librosa
from datetime import datetime
import pandas as pd
import os


class BioInspiredAudioGenerator:
    def __init__(self, sample_rate=44100, duration=10, 
                species = "species",
                individual = "individual",
                base_freq_range=(500, 2000), 
                bout_duration_range=(0.5, 1.5),
                sleep_duration_range=(1.0, 3.0),
                call_duration_range=(0.05, 0.15),
                call_spacing_range=(0.1, 0.3),
                wiener_entropy_threshold=0.2,
                euclidean_distance_threshold=0.5,
                harmonic_count_range=(2, 5),
                harmonic_decay_range=(0.3, 0.7),
                noise_variance=0.001,
                noise_floor_db=-60,  # NEW PARAMETER: Noise floor level in dB
                spectral_tilt=-6,
                formant_params=None,
                use_dynamic_formants=True):
        """
        Initialize the bio-inspired audio generator.
        
        Parameters:
        - sample_rate: Audio sample rate in Hz
        - duration: Total duration of the audio in seconds
        - species: Name of synthetic species
        - individual: Name of synthetic individual
        - base_freq_range: Range for base frequencies (Hz)
        - bout_duration_range: Range for duration of consistent call bouts (seconds)
        - sleep_duration_range: Range for sleep time between bouts (seconds)
        - call_duration_range: Range for duration of individual calls (seconds)
        - call_spacing_range: Range for time between calls within a bout (seconds)
        - wiener_entropy_threshold: Maximum allowed change in Wiener entropy
        - euclidean_distance_threshold: Maximum allowed Euclidean distance between consecutive STFT frames
        - harmonic_count_range: Range for number of harmonics to include
        - harmonic_decay_range: Range for decay factor of harmonic amplitudes
        - noise_variance: Variance of the added Gaussian white noise
        - noise_floor_db: Noise floor level in dB relative to peak (e.g., -60 dB)
        - spectral_tilt: Spectral tilt in dB/octave for natural sound (negative values)
        - formant_params: List of tuples (center_freq, bandwidth, gain) for formant filters
        - use_dynamic_formants: Whether to adjust formants based on base_freq_range
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.species = species
        self.individual = individual
        self.base_freq_range = base_freq_range
        self.bout_duration_range = bout_duration_range
        self.sleep_duration_range = sleep_duration_range
        self.call_duration_range = call_duration_range
        self.call_spacing_range = call_spacing_range
        self.wiener_entropy_threshold = wiener_entropy_threshold
        self.euclidean_distance_threshold = euclidean_distance_threshold
        self.harmonic_count_range = harmonic_count_range
        self.harmonic_decay_range = harmonic_decay_range
        self.noise_variance = noise_variance
        self.noise_floor_db = noise_floor_db  # NEW: Store noise floor parameter
        self.spectral_tilt = spectral_tilt
        self.use_dynamic_formants = use_dynamic_formants
        
        # Based on VoCallBase Standard
        self.annotations = {
            "onset" : [], 
            "offset" : [],
            "duration" : [], 
            "minFrequency" : [], 
            "maxFrequency" : [], 
            "species" : [], 
            "individual" : [],
            "filename" : [],
            "channelIndex" : []
        }

        self.project_annotations = {
            "onset" : [], 
            "offset" : [],
            "duration" : [], 
            "minFrequency" : [], 
            "maxFrequency" : [], 
            "species" : [], 
            "individual" : [],
            "filename" : [],
            "channelIndex" : []
        }
        
        # CRITICAL CHANGE: Make formant parameters dependent on frequency range
        if formant_params is None:
            if use_dynamic_formants:
                # Calculate formants based on the base frequency range
                base_freq_mid = (base_freq_range[0] + base_freq_range[1]) / 2
                
                # Format: [(center_freq, bandwidth, gain), ...]
                self.formant_params = [
                    (base_freq_mid, base_freq_mid * 0.2, 1.0),                      # Primary formant
                    (base_freq_mid * 2, base_freq_mid * 0.3, 0.5),                  # Second harmonic
                    (base_freq_mid * 3, base_freq_mid * 0.4, 0.25)                  # Third harmonic
                ]
        else:
            self.formant_params = formant_params
        
        # STFT parameters
        self.n_fft = 512
        self.hop_length = 256
        
        # Initialize empty audio buffer
        self.total_samples = int(duration * sample_rate)
        self.audio = np.zeros(self.total_samples)
        
        # Current position in samples
        self.current_sample = 0
        
        # For tracking previous STFT frames to maintain continuity
        self.prev_stft_frame = None
        
    def _wiener_entropy(self, power_spectrum):
        """Calculate Wiener entropy of a power spectrum."""
        if np.sum(power_spectrum) == 0:
            return 0
        
        # Normalize the power spectrum
        norm_power = power_spectrum / np.sum(power_spectrum)
        
        # Entropy calculation (avoid log(0))
        entropy = 0
        for p in norm_power:
            if p > 0:
                entropy -= p * np.log(p)
        
        return entropy
    
    def _euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors."""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))
    
    def _generate_call_envelope(self, duration_samples):
        """Generate amplitude envelope for a single call with stronger tapering."""
        # Create a smooth envelope with attack and release
        attack_ratio = 0.15   # Shorter attack for cleaner onset
        release_ratio = 0.25  # Longer release for natural decay
        
        attack_samples = int(duration_samples * attack_ratio)
        release_samples = int(duration_samples * release_ratio)
        sustain_samples = max(1, duration_samples - attack_samples - release_samples)
        
        # Use smoother curves for more natural sound
        attack = np.sin(np.linspace(0, np.pi/2, attack_samples))**2  # Smooth attack
        sustain = np.ones(sustain_samples)
        release = np.cos(np.linspace(0, np.pi/2, release_samples))**2  # Smooth release
        
        envelope = np.concatenate([attack, sustain, release])
        
        # Add some natural variability to the envelope (reduced amount)
        if len(envelope) > 10:  # Only add variability for longer calls
            envelope = envelope * (1 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, len(envelope))))
        
        envelope = np.clip(envelope, 0, 1)
        
        return envelope
    
    def _apply_spectral_tilt(self, spectrum, tilt_db_per_octave):
        """Apply spectral tilt to make sound less harsh."""
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        # Convert tilt to linear scale per frequency
        tilt_factor = 10**(tilt_db_per_octave/20)  # dB to amplitude ratio
        
        # Create a filter that decreases by tilt_factor for each doubling of frequency
        tilt_filter = np.ones_like(freqs)
        reference_freq = freqs[1] if len(freqs) > 1 else 1.0  # First non-DC frequency
        
        for i, f in enumerate(freqs):
            if f > 0:  # Skip DC component
                octaves_above_reference = np.log2(f / reference_freq)
                tilt_filter[i] = tilt_factor ** octaves_above_reference
        
        # Apply the tilt filter
        tilted_spectrum = spectrum * tilt_filter
        
        return tilted_spectrum
    
    def _apply_formant_based_filtering(self, audio, base_freq):
        """Apply formant-based filtering to constrain frequency content to realistic ranges."""
        if len(audio) == 0:
            return audio
        
        # Start with silence
        filtered_audio = np.zeros_like(audio)
        
        # Apply each formant as a bandpass filter
        for center_freq, bandwidth, gain in self.formant_params:
            nyquist = self.sample_rate / 2
            
            # Ensure frequencies are in valid range
            low_freq = max(20, center_freq - bandwidth/2)
            high_freq = min(nyquist * 0.95, center_freq + bandwidth/2)
            
            # Normalize for butter filter
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # Skip if the filter would be invalid
            if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1:
                continue
            
            try:
                # Design bandpass filter for this formant
                order = 4  # Higher order for better selectivity
                b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
                
                # Apply filter to the original audio
                formant_component = signal.lfilter(b, a, audio)
                
                # Add this formant component with its gain
                filtered_audio += gain * formant_component
                
            except Exception as e:
                print(f"Error creating formant filter at {center_freq} Hz: {e}")
        
        # Normalize the result
        if np.max(np.abs(filtered_audio)) > 0:
            filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))
        
        return filtered_audio

    def _apply_formants(self, audio):
        """Apply formant filters to add resonance characteristics (legacy method for compatibility)."""
        return self._apply_formant_based_filtering(audio, None)
    
    def _add_micro_variations(self, audio, variation_rate=0.01, variation_depth=0.05):
        """Add micro-variations to the amplitude to make it sound more natural."""
        if len(audio) == 0:
            return audio
            
        # Generate a slow random walk for micro variations
        num_samples = len(audio)
        num_variations = max(2, int(num_samples * variation_rate))
        variation_points = np.random.normal(0, 1, num_variations)
        
        # Interpolate to get smooth variations for all samples
        interp_indices = np.linspace(0, num_variations-1, num_samples)
        smooth_variations = np.interp(interp_indices, np.arange(num_variations), variation_points)
        
        # Apply variations to audio
        varied_audio = audio * (1 + variation_depth * smooth_variations)
        
        return varied_audio

    def _generate_call_with_continuity(self, duration, base_freq, prev_frame=None):
        """
        Generate a single call with frequency domain continuity and proper silence boundaries.
        """
        duration_samples = int(duration * self.sample_rate)
        if duration_samples <= 0:
            return np.array([]), None
            
        call_audio = np.zeros(duration_samples)
        
        # Number of STFT frames for this call
        n_frames = max(1, int(np.ceil(duration_samples / self.hop_length)))
        
        # Generate overlapping STFT frames
        stft_frames = []
        current_frame = prev_frame
        
        # Random frequency modulation parameters
        freq_mod_rate = np.random.uniform(8, 15)  # Hz
        freq_mod_depth = np.random.uniform(0.02, 0.05)  # Relative to base_freq
        
        # Random amplitude modulation parameters
        amp_mod_rate = np.random.uniform(20, 40)  # Hz
        amp_mod_depth = np.random.uniform(0.1, 0.2)
        
        # Number of harmonics for this call
        n_harmonics = np.random.randint(*self.harmonic_count_range)
        harmonic_decay = np.random.uniform(*self.harmonic_decay_range)
        
        # Generate each STFT frame with continuity constraints
        for i in range(n_frames):
            if current_frame is None:
                # First frame ever - create from scratch
                frame = np.zeros(self.n_fft // 2 + 1, dtype=complex)
                
                # Add fundamental frequency with some bandwidth
                freq_bin = int(base_freq * self.n_fft / self.sample_rate)
                
                # Ensure freq_bin is valid
                freq_bin = max(1, min(freq_bin, self.n_fft // 2 - 2))
                
                # Keep the original 3-bin approach, but ensure it's valid
                start_bin = max(0, freq_bin-1)
                end_bin = min(len(frame), freq_bin+2)
                if end_bin > start_bin:
                    frame[start_bin:end_bin] = [0.3, 1.0, 0.3][:end_bin-start_bin]
                
                # Add harmonics with decay and bandwidth
                for h in range(2, n_harmonics + 2):
                    harmonic_bin = int(h * base_freq * self.n_fft / self.sample_rate)
                    
                    if harmonic_bin > 0 and harmonic_bin < len(frame) - 1:
                        harmonic_amp = (1.0 / h) ** harmonic_decay
                        start_bin = max(0, harmonic_bin-1)
                        end_bin = min(len(frame), harmonic_bin+2)
                        if end_bin > start_bin:
                            harmonic_values = [0.3 * harmonic_amp, harmonic_amp, 0.3 * harmonic_amp]
                            frame[start_bin:end_bin] = harmonic_values[:end_bin-start_bin]
                
                # Add random phase
                phase = np.random.uniform(0, 2 * np.pi, size=frame.shape)
                frame = np.abs(frame) * np.exp(1j * phase)
                
                # Apply spectral tilt to reduce metallic sound
                frame = self._apply_spectral_tilt(frame, self.spectral_tilt)
                
            else:
                # Create frame with continuity from previous frame
                frame = current_frame.copy()
                
                # Calculate time position for modulations
                t = i * self.hop_length / self.sample_rate
                
                # Apply frequency modulation
                freq_shift = int(freq_mod_depth * base_freq * np.sin(2 * np.pi * freq_mod_rate * t) * self.n_fft / self.sample_rate)
                
                if freq_shift != 0:
                    # Shift the spectrum
                    new_frame = np.zeros_like(frame)
                    if freq_shift > 0 and freq_shift < len(frame):
                        new_frame[freq_shift:] = frame[:-freq_shift]
                    elif freq_shift < 0 and abs(freq_shift) < len(frame):
                        new_frame[:freq_shift] = frame[abs(freq_shift):]
                    frame = new_frame
                
                # Apply amplitude modulation
                amp_mod = 1.0 + amp_mod_depth * np.sin(2 * np.pi * amp_mod_rate * t)
                frame = frame * amp_mod
                
                # Apply spectral tilt
                frame = self._apply_spectral_tilt(frame, self.spectral_tilt)
                
                # Ensure continuity with Wiener entropy constraint
                candidate_frames = []
                for _ in range(10):  # Try multiple candidates
                    candidate = frame.copy()
                    
                    # Add small random changes
                    noise = np.random.normal(0, 0.05, size=len(candidate))
                    candidate = candidate + noise
                    
                    # Calculate entropy difference
                    prev_entropy = self._wiener_entropy(np.abs(current_frame) ** 2)
                    candidate_entropy = self._wiener_entropy(np.abs(candidate) ** 2)
                    entropy_diff = abs(candidate_entropy - prev_entropy)
                    
                    # Calculate Euclidean distance
                    distance = self._euclidean_distance(np.abs(current_frame), np.abs(candidate))
                    
                    # Store candidate with its scores
                    candidate_frames.append((candidate, entropy_diff, distance))
                
                # Select the best candidate frame that meets our criteria
                best_frame = None
                for candidate, entropy_diff, distance in candidate_frames:
                    if (entropy_diff < self.wiener_entropy_threshold and 
                        distance < self.euclidean_distance_threshold):
                        best_frame = candidate
                        break
                
                # If no candidate meets criteria, use the one with smallest entropy difference
                if best_frame is None:
                    best_frame = min(candidate_frames, key=lambda x: x[1])[0]
                
                frame = best_frame
            
            stft_frames.append(frame)
            current_frame = frame
        
        # Convert STFT frames back to audio
        for i, frame in enumerate(stft_frames):
            # Create full STFT frame with conjugate symmetry for real output
            full_frame = np.zeros(self.n_fft, dtype=complex)
            full_frame[:len(frame)] = frame
            
            # Proper conjugate symmetry
            if len(frame) > 1:
                conjugate_part = np.conj(frame[1:])[::-1]
                end_idx = self.n_fft - len(conjugate_part)
                if end_idx > len(frame):
                    full_frame[end_idx:] = conjugate_part
            
            # Inverse FFT
            time_segment = np.real(ifft(full_frame))
            
            # Apply a window
            window = signal.windows.hann(len(time_segment))
            time_segment = time_segment * window
            
            # Add to audio with overlap
            start_idx = i * self.hop_length
            end_idx = min(start_idx + len(time_segment), len(call_audio))
            segment_len = end_idx - start_idx
            
            if segment_len > 0:
                call_audio[start_idx:end_idx] += time_segment[:segment_len]
        
        envelope = self._generate_call_envelope(len(call_audio))
        call_audio = call_audio * envelope
        
        # Add micro variations to amplitude
        call_audio = self._add_micro_variations(call_audio)
        
        # Normalize
        if np.max(np.abs(call_audio)) > 0:
            call_audio = call_audio / np.max(np.abs(call_audio))
        
        return call_audio, current_frame
        
    def generate_audio(self):
        """Generate the complete bio-inspired audio sequence with proper silence."""
        current_time = 0.0
        
        # Keep track of the last STFT frame for continuity
        last_stft_frame = None
        
        # Initialize with silence
        self.audio.fill(0.0)
        
        while current_time < self.duration:
            # Determine if we should have a bout or sleep
            if np.random.random() < 0.7:  # 70% chance of a bout
                # Generate a bout of calls
                bout_duration = np.random.uniform(*self.bout_duration_range)
                bout_end_time = min(current_time + bout_duration, self.duration)
                
                # Use the full frequency range as specified
                bout_base_freq = np.random.uniform(*self.base_freq_range)
                
                # Get harmonic count for annotations
                n_harmonics = np.random.randint(*self.harmonic_count_range)
                
                call_time = current_time
                while call_time < bout_end_time:
                    # Determine call parameters
                    call_duration = np.random.uniform(*self.call_duration_range)
                    call_spacing = np.random.uniform(*self.call_spacing_range)
                    
                    # Make sure we don't exceed the total duration
                    if call_time + call_duration > self.duration:
                        break
                    
                    # Vary frequency slightly for each call
                    call_freq = bout_base_freq * (1 + np.random.uniform(-0.05, 0.05))
                    
                    # Generate the call
                    call_samples, last_stft_frame = self._generate_call_with_continuity(
                        call_duration, call_freq, last_stft_frame)
                    
                    if len(call_samples) == 0:
                        call_time += call_duration + call_spacing
                        continue
                    
                    # Vary amplitude between calls
                    call_amplitude = np.random.uniform(0.7, 1.0)
                    call_samples = call_samples * call_amplitude
                    
                    # CRITICAL: Replace audio in buffer instead of adding
                    start_sample = int(call_time * self.sample_rate)
                    end_sample = min(start_sample + len(call_samples), self.total_samples)
                    actual_length = end_sample - start_sample
                    
                    if actual_length > 0:
                        # Clear the region first to ensure silence between calls
                        self.audio[start_sample:end_sample] = 0.0
                        # Then add the call
                        self.audio[start_sample:end_sample] = call_samples[:actual_length]
                        
                        # Update annotations with actual times
                        actual_call_duration = actual_length / self.sample_rate
                        self.annotations["onset"].append(start_sample/self.sample_rate)
                        self.annotations["offset"].append(end_sample/self.sample_rate)
                        self.annotations["duration"].append(actual_call_duration)
                        self.annotations["minFrequency"].append(call_freq)
                        self.annotations["maxFrequency"].append(call_freq * (n_harmonics + 1))
                        self.annotations["species"].append(self.species)
                        self.annotations["individual"].append(self.individual)
                        self.annotations["channelIndex"].append(0)
                        self.annotations["filename"].append("")  # Will be filled later
                    
                    # Move to next call (add spacing for silence)
                    call_time += call_duration + call_spacing
                
                current_time = bout_end_time
            else:
                # Sleep period - explicit silence
                sleep_duration = np.random.uniform(*self.sleep_duration_range)
                sleep_end_time = min(current_time + sleep_duration, self.duration)
                
                # Ensure silence during sleep
                start_sample = int(current_time * self.sample_rate)
                end_sample = int(sleep_end_time * self.sample_rate)
                self.audio[start_sample:end_sample] = 0.0
                
                current_time = sleep_end_time
                # Reset STFT frame after sleep
                last_stft_frame = None
        
        # Apply formant-based filtering to constrain frequency content
        # Process each call region separately to apply appropriate filtering
        for i in range(len(self.annotations["onset"])):
            onset_sample = int(self.annotations["onset"][i] * self.sample_rate)
            offset_sample = int(self.annotations["offset"][i] * self.sample_rate)
            call_freq = self.annotations["minFrequency"][i]
            
            # Extract the call region
            if offset_sample <= len(self.audio) and onset_sample < offset_sample:
                call_region = self.audio[onset_sample:offset_sample].copy()
                
                # Apply formant-based filtering to constrain spectral energy
                filtered_call = self._apply_formant_based_filtering(call_region, call_freq)
                
                # Replace the original call with the filtered version
                self.audio[onset_sample:offset_sample] = filtered_call
        
        # MODIFIED: Add consistent noise floor across the entire audio
        # Calculate noise floor amplitude from dB
        noise_floor_amplitude = 10**(self.noise_floor_db / 20)  # Convert dB to linear amplitude
        
        # Generate noise for the entire clip
        noise = np.random.normal(0, noise_floor_amplitude, size=len(self.audio))
        
        # Add the noise floor to the entire audio
        self.audio = self.audio + noise
        
        # Optional: Add additional noise where there's already content (original behavior)
        if self.noise_variance > 0:
            content_noise = np.random.normal(0, np.sqrt(self.noise_variance), size=len(self.audio))
            # Only add content noise where there's already significant audio content
            content_mask = np.abs(self.audio - noise) > noise_floor_amplitude * 10  # 20dB above noise floor
            self.audio[content_mask] += content_noise[content_mask] * 0.1  # Much reduced noise
        
        # Normalize the final audio (but preserve the noise floor relationship)
        max_val = np.max(np.abs(self.audio))
        if max_val > 0:
            # Scale so that the loudest parts are at 95% of full scale
            # but maintain the noise floor relationship
            self.audio = 0.95 * self.audio / max_val
        
        return self.audio
    
    def save_audio(self, filename="bio_inspired_audio.wav"):
        """Save the generated audio to a WAV file."""
        wavfile.write(filename, self.sample_rate, self.audio.astype(np.float32))
        self.annotations["filename"] = [filename] * len(self.annotations["onset"])
        print(f"Audio saved to {filename}")
    
    def save_annotations(self, filename="bio_inspired_audio.csv"):
        "Save the synthetic annotations"
        pd.DataFrame.from_dict(self.annotations).to_csv(filename, index=False)

    def save_project_annotations(self, filename="bio_inspired_audio.csv"):
        "Save the synthetic annotations"
        pd.DataFrame.from_dict(self.project_annotations).to_csv(filename, index=False)

    def clear_annotations(self):
        for key in self.project_annotations.keys():
            self.project_annotations[key].extend(self.annotations[key])
        self.annotations = {
            "onset" : [], 
            "offset" : [],
            "duration" : [], 
            "minFrequency" : [], 
            "maxFrequency" : [], 
            "species" : [], 
            "individual" : [],
            "filename" : [],
            "channelIndex" : []
        }
    
    def clear_project_annotations(self):
        self.annotations = {
            "onset" : [], 
            "offset" : [],
            "duration" : [], 
            "minFrequency" : [], 
            "maxFrequency" : [], 
            "species" : [], 
            "individual" : [],
            "filename" : [],
            "channelIndex" : []
        }
        self.project_annotations = {
            "onset" : [], 
            "offset" : [],
            "duration" : [], 
            "minFrequency" : [], 
            "maxFrequency" : [], 
            "species" : [], 
            "individual" : [],
            "filename" : [],
            "channelIndex" : []
        }
    
    def plot_spectrogram(self, filename="bio_inspired_spectrogram.png", plot_annotations=False):
        """Plot and save a spectrogram of the generated audio."""
        plt.figure(figsize=(12, 8))
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(self.audio, n_fft=2048, hop_length=512)),
            ref=np.max
        )
        
        # Plot spectrogram
        librosa.display.specshow(
            D,
            sr=self.sample_rate,
            hop_length=512,
            x_axis='time',
            y_axis='log',
            cmap='viridis'
        )
        
        # Add annotation lines if requested
        if plot_annotations and len(self.annotations["onset"]) > 0:
            # Plot vertical lines for onsets (solid red)
            for onset in self.annotations["onset"]:
                plt.axvline(x=onset, color='r', linestyle='-', alpha=0.7, linewidth=1)
            
            # Plot vertical lines for offsets (dashed red)
            for offset in self.annotations["offset"]:
                plt.axvline(x=offset, color='r', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of Bio-inspired Audio')
        
        # Add a legend if annotations are plotted
        if plot_annotations and len(self.annotations["onset"]) > 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='r', linestyle='-', lw=1, label='Onset'),
                Line2D([0], [0], color='r', linestyle='--', lw=1, label='Offset')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Spectrogram saved to {filename}")