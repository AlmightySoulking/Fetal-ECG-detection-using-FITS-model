import wfdb
import os
import pyedflib
import pandas as pd
import numpy as np
FS_ORIGINAL_ADDB = 1000  # Sampling frequency for ADDB
FS_ORIGINAL_BDDB = 500   # Sampling frequency for BDDB
WINDOW_SECONDS = 10      # Default window size in seconds

class data_loader:

    def __init__(self, fs=1000.0):
        """
        Initialize the FITS model for fetal heartbeat detection
        """
        self.fs = fs # Sampling Frequency of the ECG data

    def load_addb_data(self, record_name, start_sec=200):
        """
        Loads a snippet from the ADDB dataset.
        """

        path = os.path.join('/home/almighty/Downloads/FITS-main/scripts/dataset/adda_dataset', record_name) # Replace this path with your own
        edf = pyedflib.EdfReader(f'{path}.edf')
        aecg_signal = edf.readSignal(1)
        edf._close()

        ann = wfdb.rdann(f'{path}.edf', 'qrs')
        rpeaks = ann.sample

        start_sample = start_sec * FS_ORIGINAL_ADDB
        end_sample = start_sample + (WINDOW_SECONDS * FS_ORIGINAL_ADDB)
        signal_snippet = aecg_signal[start_sample:end_sample]
        rpeaks_snippet = rpeaks[(rpeaks >= start_sample) & (rpeaks < end_sample)] - start_sample

        return signal_snippet, rpeaks_snippet, FS_ORIGINAL_ADDB

    def load_bddb_data(self, record_name, start_sec=10):
        """
        Loads a snippet from the BDDB dataset.
        """
        subject_path = f"/home/almighty/Downloads/FITS-main/scripts/dataset/bddb_dataset/B1_Pregnancy_dataset/{record_name}" # Replace this path with your own

        # Load signal using pandas
        ab_df = pd.read_csv(f"{subject_path}/B1_abSignals_01.txt", sep="\t", header=None)
        ab_signal_all_ch = ab_df.map(lambda x: float(str(x).replace(",", "."))).values.T
        aecg_signal = ab_signal_all_ch[0]  # Use first channel

        # Load fetal R-peaks
        rpeaks = np.loadtxt(f"{subject_path}/B1_Fetal_R_01.txt", dtype=int)
        if rpeaks.ndim > 1:
            rpeaks = rpeaks.flatten()

        start_sample = start_sec * FS_ORIGINAL_BDDB
        end_sample = start_sample + (WINDOW_SECONDS * FS_ORIGINAL_BDDB)
        signal_snippet = aecg_signal[start_sample:end_sample]
        rpeaks_snippet = rpeaks[(rpeaks >= start_sample) & (rpeaks < end_sample)] - start_sample

        return signal_snippet, rpeaks_snippet, FS_ORIGINAL_BDDB


    def load_ecg_data(self, file_path):
        """
        Load ECG data from addb or bddb datasets or generic file formats
        """
        # Check if this is a direct path to an ADDB or BDDB dataset
        if os.path.basename(file_path).startswith('r') and file_path.endswith('.edf'):
            # ADDB dataset
            record_name = os.path.basename(file_path).split('.')[0]
            directory = os.path.dirname(file_path)
            return self.load_addb_data(record_name=record_name) # (signal_snippet, rpeaks_snippet, FS_ORIGINAL_ADDB)
        elif 'B1_Pregnancy' in file_path:
            # BDDB dataset
            record_name = os.path.basename(file_path)
            return self.load_bddb_data(record_name=record_name) # (signal_snippet, rpeaks_snippet, FS_ORIGINAL_BDDB)
        else:
            # Trying generic file formats (like CSV)
            try:
                # Attempt to load as CSV
                data = pd.read_csv(file_path)
                return data, None, self.fs  # No ground truth peaks
            except:
                try:
                    # Try loading as MAT file
                    from scipy.io import loadmat
                    mat = loadmat(file_path)
                    # Adjust the key based on the actual structure
                    for key in mat.keys():
                        if isinstance(mat[key], np.ndarray) and mat[key].size > 100:
                            return pd.DataFrame(mat[key]), None, self.fs
                    raise ValueError("Could not find ECG data in the MAT file")
                except:
                    # Try loading as a different format
                    try:
                        data = pd.read_table(file_path, delimiter='\s+', header=None)
                        return data, None, self.fs
                    except:
                        raise ValueError(f"Could not load data from {file_path}")


if __name__ == '__main__':
    dL = data_loader(fs = 1000.0)
    #print(dL.load_addb_data('r01.edf'))
    #print(dL.load_bddb_data('B1_Pregnancy_01'))
    print(dL.load_ecg_data('r01.edf'))
    print(dL.load_ecg_data('B1_Pregnancy_01'))
