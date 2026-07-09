# preprocess.py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from src.data_ingestion import DataIngestion
from src.data_loader import AudioTransform

def preprocess_dataset():
    ingestion = DataIngestion()
    drone_files, bg_files = ingestion.get_all_audio_files()
    
    transform = AudioTransform()
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []  # будем хранить (spectrogram_path, label)
    
    for file_path in tqdm(bg_files, desc="Background"):
        spec = transform(file_path)   # torch.Tensor
        save_path = processed_dir / (file_path.stem + "_bg.npy")
        spec_np = spec.squeeze(0).numpy()   # удаляем канал -> (n_mels, time)
        np.save(save_path, spec_np)
        all_samples.append((save_path, 0))
    
    for file_path in tqdm(drone_files, desc="Drone"):
        spec = transform(file_path)
        save_path = processed_dir / (file_path.stem + "_drone.npy")
        spec_np = spec.squeeze(0).numpy()   # удаляем канал -> (n_mels, time)
        np.save(save_path, spec_np)
        all_samples.append((save_path, 1))
    
    # Сохраняем список всех samples в один файл для быстрого доступа
    np.save(processed_dir / "samples.npy", all_samples, allow_pickle=True)
    print(f"Preprocessed {len(all_samples)} samples to {processed_dir}")

if __name__ == "__main__":
    preprocess_dataset()