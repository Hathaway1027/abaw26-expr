import os
import cv2
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from transformers import WavLMModel, AutoFeatureExtractor
from tqdm import tqdm
import pickle


def get_total_frames_from_txt(audio_name, txt_dir):
   
    full_txt_name = [
        audio_name + ".txt",
        audio_name + "_left.txt",
        audio_name + "_right.txt",
    ]
    for txt_name in full_txt_name:
        txt_path = os.path.join(txt_dir, txt_name)
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                count = sum(1 for _ in f)
            return count - 1

    return None


def extract_features_in_chunks(audio_array, device):
    
    total_length = len(audio_array)
    chunk_length = CHUNK_SECONDS * SAMPLE_RATE

    all_hidden_states = []

    for start_idx in range(0, total_length, chunk_length):
        end_idx = min(start_idx + chunk_length, total_length)

        audio_chunk = audio_array[start_idx:end_idx]

        if len(audio_chunk) < SAMPLE_RATE * 0.1:
            continue

        inputs = processor(
            audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        all_hidden_states.append(hidden_states.squeeze(0).cpu())

    full_features = torch.cat(all_hidden_states, dim=0)
    return full_features


def save_vidname2audio_features(videoname2audiofeatures, save_path):
    
    with open(save_path + "_videoname2feature_audio.pkl", "wb") as f:
        pickle.dump(videoname2audiofeatures, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_audio_alignment(audio_dir, txt_dir, device, save_path=None):

    videoname2features = {}
    audio_files = sorted(os.listdir(audio_dir))

    for aud_filename in tqdm(audio_files, desc="Total Progress"):

        name, _ = os.path.splitext(aud_filename)
        audio_path = os.path.join(audio_dir, name + ".wav")

        target_frames = get_total_frames_from_txt(name, txt_dir)

        speech, _ = librosa.load(audio_path, sr=None, mono=True)

        feature_tensor = extract_features_in_chunks(speech, device)

        if feature_tensor is None:
            continue

        # [Time, 1024] -> [1, 1024, Time]
        input_for_interp = feature_tensor.t().unsqueeze(0)

        aligned_tensor = F.interpolate(
            input_for_interp,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

        final_array = aligned_tensor.squeeze(0).t().numpy()
        print(final_array.shape, target_frames)
        videoname2features[name] = final_array.astype(np.float32)
    save_vidname2audio_features(videoname2features, save_path)


# --- 运行 ---
base_audio_dir = "abaw_dataset/audio"
base_save_dir = "abaw_dataset/features/audio"
base_txt_dir = "abaw_dataset/ABAW_Annotations/EXPR_Recognition_Challenge"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-sb", help="子文件夹名")
    parser.add_argument("-g", default="0")
    args = parser.parse_args()

    device = "cuda:" + args.g
    
    CHUNK_SECONDS = 150  
    SAMPLE_RATE = 16000  

    print(f"Loading WavLM Model on {device}...")
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained(
        "models/audio-extract/wavlm"
    ).to(device)
    model.eval()

    process_audio_alignment(
        audio_dir=os.path.join(base_audio_dir, args.sb),
        txt_dir=os.path.join(base_txt_dir, args.sb),
        device=device,
        save_path=os.path.join(base_save_dir, args.sb),
    )
