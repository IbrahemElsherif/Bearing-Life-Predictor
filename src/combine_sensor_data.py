import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

def combine_sensor_data(raw_data_dir, acc_output_path, temp_output_path):
    """
    Combine sensor data from multiple bearing directories into single CSV files.
    
    Args:
        raw_data_dir: Directory containing bearing data folders
        acc_output_path: Path to save combined accelerometer data
        temp_output_path: Path to save combined temperature data
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    acc_columns = ["radial_force", "rotation_speed", "torque", "horizontal_vib", "vertical_vib"]
    temp_columns = ["radial_force", "rotation_speed", "torque"]
    
    raw_path = Path(raw_data_dir)
    all_acc_data = []
    all_temp_data = []
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(acc_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
    
    # Track statistics
    processed_bearings = 0
    skipped_acc_files = 0
    skipped_temp_files = 0
    
    for bearing_dir in raw_path.iterdir():
        if not bearing_dir.is_dir():
            continue
        
        bearing_id = bearing_dir.name
        logger.info(f"Processing bearing: {bearing_id}")
        
        # Combine accelerometer data
        acc_files = list(bearing_dir.glob("acc_*.csv"))
        if not acc_files:
            logger.warning(f"No accelerometer files found for bearing {bearing_id}")
        
        for acc_file in acc_files:
            try:
                df = pd.read_csv(acc_file, header=None)
                
                # Handle different file formats
                if df.shape[1] == 5:
                    df.columns = acc_columns
                elif df.shape[1] == 6:
                    # Remove extra column, assuming it's an index or timestamp
                    df = df.iloc[:, 1:]
                    df.columns = acc_columns
                else:
                    logger.warning(f"Unexpected acc file shape {df.shape} in {acc_file}. Skipping.")
                    skipped_acc_files += 1
                    continue
                
                # Add metadata
                df["bearing_id"] = bearing_id
                df["file_name"] = acc_file.name
                
                # Add timestamp if not present
                if "timestamp" not in df.columns:
                    df["timestamp"] = np.arange(len(df))
                
                all_acc_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {acc_file}: {str(e)}")
                skipped_acc_files += 1

        # Combine temperature data
        temp_files = list(bearing_dir.glob("temp_*.csv"))
        if not temp_files:
            logger.warning(f"No temperature files found for bearing {bearing_id}")
        
        for temp_file in temp_files:
            try:
                df = pd.read_csv(temp_file, header=None)
                
                # Handle different file formats
                if df.shape[1] == 3:
                    df.columns = temp_columns
                elif df.shape[1] == 5 or df.shape[1] == 6:
                    # Assume extra columns at the start, slice the last 3
                    df = df.iloc[:, -3:]
                    df.columns = temp_columns
                else:
                    logger.warning(f"Unexpected temp file shape {df.shape} in {temp_file}. Skipping.")
                    skipped_temp_files += 1
                    continue
                
                # Add metadata
                df["bearing_id"] = bearing_id
                df["file_name"] = temp_file.name
                
                # Add timestamp if not present
                if "timestamp" not in df.columns:
                    df["timestamp"] = np.arange(len(df))
                
                all_temp_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {temp_file}: {str(e)}")
                skipped_temp_files += 1
        
        processed_bearings += 1
    
    # Save combined CSVs
    if all_acc_data:
        acc_df = pd.concat(all_acc_data, ignore_index=True)
        acc_df.to_csv(acc_output_path, index=False)
        logger.info(f"Saved {len(acc_df)} accelerometer records to {acc_output_path}")
    else:
        logger.warning("No accelerometer data to save")
    
    if all_temp_data:
        temp_df = pd.concat(all_temp_data, ignore_index=True)
        temp_df.to_csv(temp_output_path, index=False)
        logger.info(f"Saved {len(temp_df)} temperature records to {temp_output_path}")
    else:
        logger.warning("No temperature data to save")
    
    # Log summary
    logger.info(f"Processing complete: {processed_bearings} bearings processed")
    logger.info(f"Skipped files: {skipped_acc_files} accelerometer, {skipped_temp_files} temperature")

# Example usage:
if __name__ == "__main__":
    combine_sensor_data(
        raw_data_dir="data/raw",
        acc_output_path="data/processed/all_acc_data.csv",
        temp_output_path="data/processed/all_temp_data.csv"
    )
