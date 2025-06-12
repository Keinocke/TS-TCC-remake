import os
import numpy as np
import mne
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
# ==== Configuration ====
output_root = Path("data/EpilepsyCustom")
epilepsy_dir = Path("C:/Users/filip/OneDrive/Skrivebord/Tuh_eeg_data/tuh_eeg_epilepsy/v2.0.1/00_epilepsy")
no_epilepsy_dir = Path("C:/Users/filip/OneDrive/Skrivebord/Tuh_eeg_data/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy")
fs = 256
duration = 10  # seconds to extract per sample
n_channels = 19  # change based on actual data
example_file = list(epilepsy_dir.rglob("*.edf"))[0]  # Get an example file to read the channel names

#------------------- Channel list ----------------
def get_channel_list(example_edf):
    raw = mne.io.read_raw_edf(example_edf, preload=False, verbose=False)
    print("All channels:", raw.ch_names)

    standard_10_20 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                      'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                      'Fz', 'Cz', 'Pz']

    cleaned = []
    for ch in raw.ch_names:
        if ch.startswith('EEG '):
            name = ch.replace('EEG ', '').replace('-LE', '').replace('-Ref', '').strip().upper()
            if name in [c.upper() for c in standard_10_20]:
                cleaned.append(name.capitalize())
    print("Cleaned channels:", cleaned)

    return cleaned

#call funktion
channel_list = get_channel_list(example_file)

#-----------------------------------------------------




# ==== Output folders ====
splits = ['train', 'val', 'test']
classes = ['class0', 'class1']  # 0 = no epilepsy, 1 = epilepsy
for split in splits:
    for cls in classes:
        Path(output_root / split / cls).mkdir(parents=True, exist_ok=True)

# ==== Collect files ====
def get_edf_files(base_dir, label):
    files = list(Path(base_dir).rglob("*.edf"))
    return [(f, label) for f in files]

epileptic = get_edf_files(epilepsy_dir, 1)
non_epileptic = get_edf_files(no_epilepsy_dir, 0)

# === Select 100 from each ===
epileptic = epileptic[:100]
non_epileptic = non_epileptic[:100]
all_data = epileptic + non_epileptic
np.random.seed(42)
np.random.shuffle(all_data)

# === Split ===
train, temp = train_test_split(all_data, test_size=0.4, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

splits_data = {'train': train, 'val': val, 'test': test}

#------------- Process and save data with 19 channels ------------
# def edf_to_npy(edf_path):
#     try:
#         raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
#         raw.resample(fs)
#         data = raw.get_data()
#         if data.shape[0] > n_channels:
#             data = data[:n_channels]
#         segment_len = fs * duration
#         if data.shape[1] >= segment_len:
#             return data[:, :segment_len]
#         elif data.shape[1] > 0:
#             padded = np.zeros((n_channels, segment_len))
#             padded[:, :data.shape[1]] = data
#             return padded
#         else:
#             return None  # Skip files with empty data
#     except Exception as e:
#         print(f" Error reading {edf_path}: {e}")
#         return None
#------------------------------------------------

#-------------------- Process and save data with 1 channel ----------------
def edf_to_npy(edf_path, selected_channel):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.resample(fs)
        
        # Normalize expected names like "Fp1" to actual names like "EEG FP1-LE"
        matched_ch = [ch for ch in raw.ch_names if selected_channel.upper() in ch.upper() and ch.startswith("EEG")]
        if not matched_ch:
            print(f" Channel {selected_channel} not found in {edf_path.name}")
            return None

        raw.pick_channels([matched_ch[0]])  # select actual channel name
        data = raw.get_data()
        segment_len = fs * duration

        if data.shape[1] >= segment_len:
            return data[:, :segment_len]
        elif data.shape[1] > 0:
            padded = np.zeros((1, segment_len))
            padded[:, :data.shape[1]] = data
            return padded
        else:
            print(f" Empty data in {edf_path.name} for {selected_channel}")
            return None
    except Exception as e:
        print(f" Error reading {edf_path.name} for {selected_channel}: {e}")
        return None

#---------------------------------------------------------


#------------------- main loop per 19 channel ----------------
# for split, samples in splits_data.items():
#     for i, (edf_path, label) in enumerate(samples):
#         try:
#             arr = edf_to_npy(edf_path)
#             out_path = output_root / split / f"class{label}" / f"sample_{i}.npy"
#             np.save(out_path, arr.astype(np.float32), allow_pickle=True)
#         except Exception as e:
#             print(f"Failed to process {edf_path}: {e}")
#--------------------------------------------------------   

#------------------- main loop per 1 channel ----------------
for selected_channel in channel_list:
    print(f"\n Processing channel: {selected_channel}")
    output_root_channel = Path(f"data/EpilepsyCustom_{selected_channel}")
    
    for split in splits:
        for cls in classes:
            Path(output_root_channel / split / cls).mkdir(parents=True, exist_ok=True)

    for split, samples in splits_data.items():
        for i, (edf_path, label) in enumerate(samples):
            try:
                arr = edf_to_npy(edf_path, selected_channel)
                if arr is None:
                    continue
                out_path = output_root_channel / split / f"class{label}" / f"sample_{i}.npy"
                np.save(out_path, arr.astype(np.float32), allow_pickle=True)
            except Exception as e:
                print(f"Failed to process {edf_path}: {e}")
#--------------------------------------------------------