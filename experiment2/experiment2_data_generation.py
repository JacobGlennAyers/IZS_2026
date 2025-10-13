from BioInspiredAudioGenerator import *
import librosa
from datetime import datetime
import os
import numpy as np
from joblib import Parallel, delayed
import random
import copy

def reset_seeds(val=42):
    np.random.seed(val)
    random.seed(val)

def process_species(species_params, output_path, hours, species_index, seed=42):
    """Process a single species, generating multiple audio files."""
    if seed is not None:
        reset_seeds(seed)
    generator = BioInspiredAudioGenerator(**species_params)
    cur_dir = os.path.join(output_path, f"species_{species_index}")
    os.makedirs(cur_dir, exist_ok=True)
    
    for hour in range(hours):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.wav"
        cur_audio = generator.generate_audio()
        generator.save_audio(os.path.join(cur_dir, filename))
        generator.clear_annotations()
        generator.save_project_annotations(os.path.join(output_path, f'species_{species_index}.csv'))
    generator.clear_project_annotations()
    
    return species_index

if __name__ == "__main__":
    reset_seeds()
    
    # Enhanced base parameters for more realistic bioacoustic data
    base_params = {
        "sample_rate": 16_000,
        "duration": 60,
        "individual": "none", 
        "sleep_duration_range": (0.8, 2.0),
        "call_duration_range": (0.5, 1.00),
        "wiener_entropy_threshold": 0.1,
        "euclidean_distance_threshold": 0.15,
        "harmonic_count_range": (3, 6),
        "harmonic_decay_range": (0.4, 0.6),
        "noise_variance": 0.001,
        "spectral_tilt": -8,
        "formant_params": None,
        "use_dynamic_formants": False
    }
    
    # Keep original frequency centers as specified
    base_freq_centers = [500, 650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850]
    
    species_list = []
    
    for i in range(10):
        # CRITICAL FIX: Deep copy for each species to prevent parameter bleeding
        species_params = copy.deepcopy(base_params)
        species_params["species"] = f"species_{i}"
        
        # 1. FREQUENCY COMPLEXITY - Use your original approach
        center_freq = base_freq_centers[i]
        # Keep your original bandwidth approach
        range_width = 300 + (i * 80)  # Your original progression
        
        current_base_freq_min = center_freq - (range_width / 2)
        current_base_freq_max = center_freq + (range_width / 2)
        species_params["base_freq_range"] = (current_base_freq_min, current_base_freq_max)
        
        # 2. TEMPORAL COMPLEXITY - More varied and complex timing patterns
        # Bout duration: shorter more frequent bouts -> longer more complex bouts
        bout_min = max(1.0, 3.0 - i * 0.2)  # Minimum decreases slightly
        bout_max = 5.0 + i * 1.5  # Maximum increases significantly
        species_params["bout_duration_range"] = (bout_min, bout_max)
        
        # Call spacing: more irregular for complex species
        spacing_min = max(0.3, 1.5 - i * 0.1)  # Tighter minimum spacing
        spacing_max = 1.8 + i * 0.3  # Much more variable maximum
        species_params["call_spacing_range"] = (spacing_min, spacing_max)
        
        # Call duration: more variable for complex species
        call_min = max(0.2, 0.5 - i * 0.03)  # Slightly shorter minimum
        call_max = 1.0 + i * 0.2  # Longer maximum calls
        species_params["call_duration_range"] = (call_min, call_max)
        
        # 3. SPECTRAL COMPLEXITY - Progressive harmonic richness
        # More harmonics and more varied decay patterns
        harmonic_min = 3
        harmonic_max = min(12, 6 + i * 2)  # Up to 12 harmonics for most complex
        species_params["harmonic_count_range"] = (harmonic_min, harmonic_max)
        
        # More varied harmonic decay for complex species
        decay_min = max(0.2, 0.4 - i * 0.02)
        decay_max = min(0.9, 0.6 + i * 0.05)
        species_params["harmonic_decay_range"] = (decay_min, decay_max)
        
        # 4. CONTINUITY THRESHOLDS - Allow more variation for complex species
        # Higher thresholds = more spectral and temporal variation allowed
        wiener_threshold = 0.2 + i * 0.1  # More entropy variation
        euclidean_threshold = 0.2 + i * 0.1  # More spectral distance variation
        
        species_params["wiener_entropy_threshold"] = min(wiener_threshold, 0.6)
        species_params["euclidean_distance_threshold"] = min(euclidean_threshold, 0.5)
        
        # 5. NOISE AND REALISM
        #
        noise_var = 0.01 #+ i * 0.005  # More noise variance
        spectral_tilt = -8 #+ i * 0.8  # Less natural (flatter) spectrum for complex species
        
        species_params["noise_variance"] = noise_var
        species_params["spectral_tilt"] = spectral_tilt
        
        # 6. FORMANT COMPLEXITY - Keep your original formant approach with Nyquist safety
        base_freq_mid = center_freq
        
        # Use your original formant design but ensure all harmonics stay under 8kHz
        formant_params = [
            (base_freq_mid, base_freq_mid * 0.2, 1.0),        # Primary formant
            (base_freq_mid * 2, base_freq_mid * 0.3, 0.5),    # Second harmonic
            (base_freq_mid * 3, base_freq_mid * 0.4, 0.25)    # Third harmonic
        ]
        
        # Filter out any formants that would exceed Nyquist limit
        safe_formant_params = []
        for formant_center, formant_bandwidth, formant_gain in formant_params:
            # Check if formant center + bandwidth/2 stays under 8kHz
            if formant_center + (formant_bandwidth / 2) < 8000:
                safe_formant_params.append((formant_center, formant_bandwidth, formant_gain))
            else:
                # If the formant would exceed Nyquist, skip it
                print(f"  Skipping formant at {formant_center}Hz for species {i} (would exceed Nyquist)")
        
        species_params["formant_params"] = safe_formant_params
        
        # 7. SLEEP PATTERN COMPLEXITY - More varied inactive periods
        sleep_min = max(0.5, 0.8 - i * 0.05)  # Slightly shorter minimum
        sleep_max = 2.0 + i * 0.5  # Much longer potential silences
        species_params["sleep_duration_range"] = (sleep_min, sleep_max)
        
        species_list.append(species_params)
    
    # Print parameter verification - This helps ensure monotonic increase
    print("COMPLEXITY PROGRESSION VERIFICATION:")
    print("="*60)
    for i, params in enumerate(species_list):
        freq_range = params['base_freq_range']
        freq_bandwidth = freq_range[1] - freq_range[0]
        bout_variance = params['bout_duration_range'][1] - params['bout_duration_range'][0]
        call_variance = params['call_spacing_range'][1] - params['call_spacing_range'][0]
        max_harmonics = params['harmonic_count_range'][1]
        wiener_thresh = params['wiener_entropy_threshold']
        
        print(f"Species {i:2d}: "
              f"Freq_BW={freq_bandwidth:4.0f}Hz, "
              f"Bout_Var={bout_variance:.1f}s, "
              f"Call_Var={call_variance:.2f}s, "
              f"Max_Harm={max_harmonics:2d}, "
              f"Wiener={wiener_thresh:.3f}")
    
    # Verify monotonic increases in key complexity measures
    print("\nMONOTONIC VERIFICATION:")
    print("-" * 30)
    
    # Check frequency bandwidth
    freq_bandwidths = [(p['base_freq_range'][1] - p['base_freq_range'][0]) for p in species_list]
    freq_monotonic = all(freq_bandwidths[i] <= freq_bandwidths[i+1] for i in range(len(freq_bandwidths)-1))
    print(f"Frequency bandwidth monotonic: {'✓' if freq_monotonic else '✗'}")
    
    # Check bout duration variance  
    bout_variances = [(p['bout_duration_range'][1] - p['bout_duration_range'][0]) for p in species_list]
    bout_monotonic = all(bout_variances[i] <= bout_variances[i+1] for i in range(len(bout_variances)-1))
    print(f"Bout duration variance monotonic: {'✓' if bout_monotonic else '✗'}")
    
    # Check harmonic count
    max_harmonics = [p['harmonic_count_range'][1] for p in species_list]
    harmonic_monotonic = all(max_harmonics[i] <= max_harmonics[i+1] for i in range(len(max_harmonics)-1))
    print(f"Max harmonic count monotonic: {'✓' if harmonic_monotonic else '✗'}")
    
    # Check Wiener threshold (complexity allowance)
    wiener_thresholds = [p['wiener_entropy_threshold'] for p in species_list]
    wiener_monotonic = all(wiener_thresholds[i] <= wiener_thresholds[i+1] for i in range(len(wiener_thresholds)-1))
    print(f"Wiener threshold monotonic: {'✓' if wiener_monotonic else '✗'}")
    
    if not (freq_monotonic and bout_monotonic and harmonic_monotonic and wiener_monotonic):
        print("\n⚠️  WARNING: Some complexity measures are not monotonic!")
        print("Consider adjusting the parameter progression.")
    else:
        print("\n✅ All key complexity measures increase monotonically!")
    
    # Generate the audio files
    output_path = "data"
    clip_count = 600  # Number of clips per species
    #clip_count = 1200
    
    os.makedirs(output_path, exist_ok=True)
    seeds = [42 + i for i in range(len(species_list))]
    
    print(f"\nGenerating audio for {len(species_list)} species...")
    results = Parallel(n_jobs=10, verbose=10)(
        delayed(process_species)(
            species_params, output_path, clip_count, idx, seed=seeds[idx]
        ) for idx, species_params in enumerate(species_list)
    )
    
    print(f"\n✅ Completed processing {len(results)} species")
    print(f"Audio files saved to: {output_path}")
    print("\nYou can now test your complexity measure on this monotonically increasing dataset!")