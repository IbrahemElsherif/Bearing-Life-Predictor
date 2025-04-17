import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch
from scipy.fft import fft
from base import Base
import logging

class FeatureEngineering(Base):
    def __init__(self):
        super().__init__()
        self.raw_data_dir = self.config["paths"]["raw_data_dir"]
        self.processed_data_dir = self.config["paths"]["processed_data_dir"]
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_rul(self, bearing_data, failure_times):
        """Calculate Remaining Useful Life for each bearing data point"""
        # Get bearing ID
        bearing_id = bearing_data['bearing_id'].iloc[0]
        
        # Get failure time for this bearing
        if bearing_id in failure_times:
            total_lifetime = failure_times[bearing_id]
            
            # Add timestamp if not present
            if 'timestamp' not in bearing_data.columns:
                bearing_data['timestamp'] = np.arange(len(bearing_data))
                
            # Calculate RUL for each timestamp
            max_timestamp = bearing_data['timestamp'].max()
            bearing_data['rul'] = total_lifetime - bearing_data['timestamp']
            
            # Ensure RUL is non-negative
            bearing_data['rul'] = bearing_data['rul'].clip(lower=0)
            
            # Normalize RUL to 0-1 range
            scaler = MinMaxScaler()
            bearing_data['rul_normalized'] = scaler.fit_transform(
                bearing_data['rul'].values.reshape(-1, 1)
            )
            
        else:
            # For test bearings without known failure time
            bearing_data['rul'] = -1  # Unknown
            bearing_data['rul_normalized'] = -1
            
        return bearing_data
    
    def extract_time_domain_features(self, signal):
        """Extract time domain features from a signal"""
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        features['rms'] = np.sqrt(np.mean(np.square(signal)))
        features['kurtosis'] = pd.Series(signal).kurtosis()
        features['skewness'] = pd.Series(signal).skew()
        features['peak_to_peak'] = features['max'] - features['min']
        features['crest_factor'] = features['max'] / features['rms'] if features['rms'] > 0 else 0
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
        features['impulse_factor'] = features['max'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
        
        return features
    
    def extract_frequency_domain_features(self, signal, fs=25600):
        """Extract frequency domain features from a signal"""
        features = {}
        
        # FFT
        signal_fft = fft(signal)
        n = len(signal)
        freq = np.fft.fftfreq(n, d=1/fs)
        magnitude = np.abs(signal_fft)
        
        # Only consider positive frequencies
        pos_mask = freq > 0
        freq = freq[pos_mask]
        magnitude = magnitude[pos_mask]
        
        if len(magnitude) > 0:
            # Features
            features['peak_freq'] = freq[np.argmax(magnitude)]
            features['freq_mean'] = np.mean(magnitude)
            features['freq_std'] = np.std(magnitude)
            features['freq_skewness'] = pd.Series(magnitude).skew()
            features['freq_kurtosis'] = pd.Series(magnitude).kurtosis()
            
            # Power spectral density
            f, psd = welch(signal, fs=fs)
            features['psd_max'] = np.max(psd)
            features['psd_sum'] = np.sum(psd)
            
            # Frequency bands energy
            if len(f) > 10:  # Ensure we have enough frequency points
                # Define frequency bands (adjust based on domain knowledge)
                bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
                
                for i, (low, high) in enumerate(bands):
                    band_mask = (f >= low) & (f < high)
                    if np.any(band_mask):
                        features[f'band_{i}_energy'] = np.sum(psd[band_mask])
                    else:
                        features[f'band_{i}_energy'] = 0
        else:
            # Default values if no valid frequencies
            features['peak_freq'] = 0
            features['freq_mean'] = 0
            features['freq_std'] = 0
            features['freq_skewness'] = 0
            features['freq_kurtosis'] = 0
            features['psd_max'] = 0
            features['psd_sum'] = 0
            for i in range(5):
                features[f'band_{i}_energy'] = 0
        
        return features
    
    def extract_condition_from_bearing_id(self, bearing_id):
        """Extract condition information from bearing ID"""
        # Format: BearingX_Y where X is the bearing number and Y is the condition
        parts = bearing_id.split('_')
        if len(parts) == 2:
            return int(parts[1])
        return 0
    
    def process_bearing_data(self, bearing_id, failure_times=None):
        """Process data for a single bearing"""
        bearing_dir = os.path.join(self.raw_data_dir, bearing_id)
        if not os.path.exists(bearing_dir):
            self.logger.warning(f"Directory not found for bearing {bearing_id}")
            return None
        
        # Load accelerometer data
        acc_files = [f for f in os.listdir(bearing_dir) if f.startswith('acc_') and f.endswith('.csv')]
        if not acc_files:
            self.logger.warning(f"No accelerometer files found for bearing {bearing_id}")
            return None
        
        all_features = []
        
        for acc_file in acc_files:
            try:
                # Load data
                acc_path = os.path.join(bearing_dir, acc_file)
                acc_data = pd.read_csv(acc_path, header=None)
                
                # Handle different file formats
                if acc_data.shape[1] == 5:
                    acc_data.columns = ["radial_force", "rotation_speed", "torque", "horizontal_vib", "vertical_vib"]
                elif acc_data.shape[1] == 6:
                    acc_data = acc_data.iloc[:, 1:]
                    acc_data.columns = ["radial_force", "rotation_speed", "torque", "horizontal_vib", "vertical_vib"]
                else:
                    self.logger.warning(f"Unexpected acc file shape {acc_data.shape} in {acc_file}. Skipping.")
                    continue
                
                # Add metadata
                acc_data['bearing_id'] = bearing_id
                acc_data['file_name'] = acc_file
                acc_data['timestamp'] = np.arange(len(acc_data))
                
                # Calculate RUL if failure times are provided
                if failure_times is not None:
                    acc_data = self.calculate_rul(acc_data, failure_times)
                
                # Extract features from vibration signals
                window_size = 1000  # Adjust based on domain knowledge
                overlap = 0.5  # 50% overlap
                step = int(window_size * (1 - overlap))
                
                for i in range(0, len(acc_data) - window_size, step):
                    window = acc_data.iloc[i:i+window_size]
                    
                    # Extract features from horizontal and vertical vibration
                    h_vib = window['horizontal_vib'].values
                    v_vib = window['vertical_vib'].values
                    
                    # Time domain features
                    h_time_features = self.extract_time_domain_features(h_vib)
                    v_time_features = self.extract_time_domain_features(v_vib)
                    
                    # Frequency domain features
                    h_freq_features = self.extract_frequency_domain_features(h_vib)
                    v_freq_features = self.extract_frequency_domain_features(v_vib)
                    
                    # Combine features
                    features = {
                        'bearing_id': bearing_id,
                        'window_id': f"{acc_file}_{i}",
                        'timestamp': window['timestamp'].iloc[0],
                        'condition': self.extract_condition_from_bearing_id(bearing_id)
                    }
                    
                    # Add RUL columns if available
                    if failure_times is not None:
                        features['rul'] = window['rul'].iloc[0]
                        features['rul_normalized'] = window['rul_normalized'].iloc[0]
                    
                    # Add time domain features
                    for k, v in h_time_features.items():
                        features[f'h_{k}'] = v
                    for k, v in v_time_features.items():
                        features[f'v_{k}'] = v
                    
                    # Add frequency domain features
                    for k, v in h_freq_features.items():
                        features[f'h_{k}'] = v
                    for k, v in v_freq_features.items():
                        features[f'v_{k}'] = v
                    
                    # Add operational parameters
                    features['radial_force'] = window['radial_force'].mean()
                    features['rotation_speed'] = window['rotation_speed'].mean()
                    features['torque'] = window['torque'].mean()
                    
                    all_features.append(features)
                    
            except Exception as e:
                self.logger.error(f"Error processing {acc_file} for bearing {bearing_id}: {str(e)}")
                continue
        
        return pd.DataFrame(all_features) if all_features else None
    
    def remove_correlated_features(self, df, threshold=0.95):
        """Remove highly correlated features"""
        # Identify RUL columns to preserve
        rul_cols = [col for col in df.columns if 'rul' in col.lower()]
        
        # Select numeric columns excluding RUL columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in rul_cols]
        numeric_df = df[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Find highly correlated features
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Remove highly correlated features (excluding RUL columns)
        df = df.drop(columns=[col for col in to_drop if col not in rul_cols])
        
        return df
    
    def process_all_bearings(self):
        """Process all bearings and combine features"""
        self.logger.info("Starting feature extraction for all bearings")
        
        # Get all bearing directories
        bearing_dirs = [d for d in os.listdir(self.raw_data_dir) 
                       if os.path.isdir(os.path.join(self.raw_data_dir, d)) 
                       and d.startswith('Bearing')]
        
        if not bearing_dirs:
            self.logger.error("No bearing directories found")
            return None
        
        # First pass: collect failure times
        failure_times = {}
        for bearing_id in bearing_dirs:
            bearing_dir = os.path.join(self.raw_data_dir, bearing_id)
            acc_files = [f for f in os.listdir(bearing_dir) if f.startswith('acc_') and f.endswith('.csv')]
            
            if acc_files:
                # Get the last file
                last_file = sorted(acc_files)[-1]
                acc_path = os.path.join(bearing_dir, last_file)
                try:
                    acc_data = pd.read_csv(acc_path, header=None)
                    # Use the length of the last file as the failure time
                    failure_times[bearing_id] = len(acc_data)
                except Exception as e:
                    self.logger.error(f"Error reading {last_file} for bearing {bearing_id}: {str(e)}")
                    failure_times[bearing_id] = 1000  # Default value
        
        # Second pass: process all bearings with failure times
        all_features = []
        for bearing_id in bearing_dirs:
            self.logger.info(f"Processing bearing: {bearing_id}")
            features_df = self.process_bearing_data(bearing_id, failure_times)
            
            if features_df is not None:
                all_features.append(features_df)
        
        if not all_features:
            self.logger.error("No features extracted from any bearing")
            return None
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Remove highly correlated features
        combined_features = self.remove_correlated_features(combined_features)
        
        # Save processed features
        output_path = os.path.join(self.processed_data_dir, "processed_features.csv")
        combined_features.to_csv(output_path, index=False)
        self.logger.info(f"Processed features saved to {output_path}")
        
        return combined_features