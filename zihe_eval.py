#!/usr/bin/env python3

import os
import glob
import torch
import torchaudio
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Import existing modules
from main_fm import run_validation as run_validation_fm
from main_tm import run_validation as run_validation_tm
from models import Hubert, Wav2Vec2BERT
from utils import get_model, compute_eer
from collections import OrderedDict


class MP3Dataset(Dataset):
    """Simple dataset for MP3 files with robust mono conversion"""

    def __init__(self, mp3_dir, sample_rate=16000, max_length=64000):
        self.mp3_files = glob.glob(os.path.join(mp3_dir, "*.mp3"))
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.labels = {}

        # Create labels: 1 for original (genuine), 0 for generated
        for file_path in self.mp3_files:
            filename = os.path.basename(file_path)
            self.labels[file_path] = 1 if "original" in filename else 0

        print(f"Found {len(self.mp3_files)} MP3 files")
        genuine_count = sum(self.labels.values())
        print(
            f"Genuine samples: {genuine_count}, Generated samples: {len(self.mp3_files) - genuine_count}"
        )

    def __len__(self):
        return len(self.mp3_files)

    def __getitem__(self, idx):
        file_path = self.mp3_files[idx]

        try:
            # Load audio file
            waveform, sr = torchaudio.load(file_path)

            # FORCE MONO CONVERSION - this is critical
            if waveform.shape[0] > 1:
                # Convert stereo to mono by averaging channels
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Ensure we have exactly 1 channel
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)

            # Pad or truncate to fixed length
            if waveform.size(1) < self.max_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_length - waveform.size(1))
                )
            else:
                waveform = waveform[:, : self.max_length]

            # Remove the channel dimension to get [max_length]
            waveform = waveform.squeeze(0)

            # DOUBLE CHECK: ensure final shape is exactly [max_length]
            if waveform.shape != (self.max_length,):
                print(
                    f"WARNING: Incorrect shape {waveform.shape} for {file_path}, fixing..."
                )
                waveform = waveform.flatten()[: self.max_length]
                if len(waveform) < self.max_length:
                    waveform = torch.nn.functional.pad(
                        waveform, (0, self.max_length - len(waveform))
                    )

            label = self.labels[file_path]
            return waveform.numpy().astype(np.float32), label, file_path

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a zero tensor as fallback
            waveform = torch.zeros(self.max_length, dtype=torch.float32)
            return waveform.numpy(), 0, file_path


def safe_collate_fn(batch):
    """Ultra-safe collate function that handles any tensor shape issues"""
    waveforms, labels, file_paths = zip(*batch)

    # Convert all waveforms to tensors and ensure they're all [64000]
    processed_waveforms = []
    for i, waveform in enumerate(waveforms):
        # Convert to tensor
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = waveform.float()

        # Force to correct shape
        if waveform_tensor.shape != torch.Size([64000]):
            print(
                f"Fixing shape {waveform_tensor.shape} -> [64000] for {file_paths[i]}"
            )
            waveform_tensor = waveform_tensor.flatten()[:64000]
            if len(waveform_tensor) < 64000:
                waveform_tensor = torch.nn.functional.pad(
                    waveform_tensor, (0, 64000 - len(waveform_tensor))
                )

        processed_waveforms.append(waveform_tensor)

    # Stack all waveforms
    waveforms_batch = torch.stack(processed_waveforms)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return waveforms_batch, labels_batch, file_paths


def run_validation_fm_with_details(model, feature_extractor, data_loader, sr):
    """Modified version that returns individual predictions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs_list = []
    labels_list = []
    file_paths_list = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, names in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.numpy()
            inputs = feature_extractor(
                batch_x,
                sampling_rate=sr,
                return_attention_mask=True,
                padding_value=0,
                return_tensors="pt",
            ).to(device)
            batch_y = batch_y.to(device)
            inputs["labels"] = batch_y
            outputs = model(**inputs)

            batch_probs = outputs.logits.softmax(dim=-1)
            batch_label = batch_y.detach().to("cpu").numpy().tolist()

            outputs_list.extend(
                batch_probs[:, 1].tolist()
            )  # Confidence for class 1 (genuine)
            labels_list.extend(batch_label)
            file_paths_list.extend(names)

    # Calculate metrics
    auroc = roc_auc_score(labels_list, outputs_list)
    eer = compute_eer(np.array(labels_list), np.array(outputs_list))
    preds = (np.array(outputs_list) > eer[1]).astype(int)
    acc = np.mean(np.array(labels_list) == np.array(preds))

    return acc, auroc, eer, outputs_list, labels_list, file_paths_list, preds


def run_validation_tm_with_details(config, data_loader, model, device):
    """Modified version that returns individual predictions"""
    from audio_feature_extraction import LFCC
    from torchaudio.transforms import Spectrogram

    outputs_list = []
    labels_list = []
    file_paths_list = []

    if config["model_config"]["architecture"] == "LCNN":
        lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
    elif config["model_config"]["architecture"] == "ResNet":
        spectrogram = Spectrogram(n_fft=512)

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, names in tqdm(data_loader, desc="Evaluating"):
            if config["model_config"]["architecture"] == "LCNN":
                batch_x = torch.unsqueeze(lfcc(batch_x.float()).transpose(1, 2), 1).to(
                    device
                )
            elif config["model_config"]["architecture"] == "ResNet":
                batch_x = torch.unsqueeze(spectrogram(batch_x.float()), 1).to(device)
            else:
                batch_x = batch_x.float().to(device)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)

            batch_prob = F.softmax(batch_out, dim=1).detach().to("cpu").numpy()
            batch_label = batch_y.detach().to("cpu").numpy().tolist()

            outputs_list.extend(
                batch_prob[:, 1].tolist()
            )  # Confidence for class 1 (genuine)
            labels_list.extend(batch_label)
            file_paths_list.extend(names)

    # Calculate metrics
    auroc = roc_auc_score(labels_list, outputs_list)
    eer = compute_eer(np.array(labels_list), np.array(outputs_list))
    preds = (np.array(outputs_list) > eer[1]).astype(int)
    acc = np.mean(np.array(labels_list) == np.array(preds))

    return acc, auroc, eer, outputs_list, labels_list, file_paths_list, preds


def save_results_to_file(
    results_file,
    model_name,
    acc,
    auroc,
    eer,
    outputs_list,
    labels_list,
    file_paths_list,
    preds,
):
    """Save detailed results to file"""
    with open(results_file, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"MODEL: {model_name.upper()}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {acc:.4f}\n")
        f.write(f"  AUROC: {auroc:.4f}\n")
        f.write(f"  EER: {eer[0]:.4f}\n\n")

        f.write("Individual File Results:\n")
        f.write("Filename\tTrue_Label\tPredicted\tConfidence\tCorrect\n")
        f.write("-" * 80 + "\n")

        for i, (file_path, true_label, pred, confidence) in enumerate(
            zip(file_paths_list, labels_list, preds, outputs_list)
        ):
            filename = os.path.basename(file_path)
            correct = "✓" if true_label == pred else "✗"
            f.write(f"{filename}\t{true_label}\t{pred}\t{confidence:.4f}\t{correct}\n")
        f.write("\n")


def evaluate_foundation_models(mp3_dir, weights_dir, results_file):
    """Evaluate HuBert and Wav2Vec2BERT using main_fm functions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models to evaluate
    models_config = {
        "hubert": {
            "model_name": "facebook/hubert-large-ls960-ft",
            "weight_file": "hubert_large_wavefake.pth",
            "sampling_rate": 16000,
        },
        "wave2vec2bert": {
            "model_name": "facebook/w2v-bert-2.0",
            "weight_file": "wave2vec2bert_wavefake.pth",
            "sampling_rate": 16000,
        },
    }

    # Create dataset and dataloader with safe collate
    dataset = MP3Dataset(mp3_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Smaller batch size for safety
        shuffle=False,
        num_workers=0,  # No multiprocessing to avoid worker issues
        collate_fn=safe_collate_fn,
    )

    for model_name, config in models_config.items():
        print(f"\nEvaluating {model_name}...")

        # Initialize model
        if model_name == "hubert":
            model = Hubert(config["model_name"])
        elif model_name == "wave2vec2bert":
            model = Wav2Vec2BERT(config["model_name"])

        model = model.to(device)

        # Load weights
        weight_path = os.path.join(weights_dir, config["weight_file"])
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location=device)
            model.load_state_dict(ckpt)
            print(f"Loaded weights from {weight_path}")
        else:
            print(
                f"Warning: Weight file {weight_path} not found, using pretrained weights"
            )

        # Get feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])

        # Run evaluation with details
        try:
            acc, auroc, eer, outputs_list, labels_list, file_paths_list, preds = (
                run_validation_fm_with_details(
                    model, feature_extractor, dataloader, config["sampling_rate"]
                )
            )

            # Save results to file
            save_results_to_file(
                results_file,
                model_name,
                acc,
                auroc,
                eer,
                outputs_list,
                labels_list,
                file_paths_list,
                preds,
            )

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")


def evaluate_aasist(mp3_dir, weights_dir, results_file):
    """Evaluate AASIST using main_tm functions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load AASIST config
    config_path = "./config/AASIST.conf"
    with open(config_path, "r") as f:
        config = json.loads(f.read())

    # Update model path to weights directory
    config["model_path"] = os.path.join(weights_dir, "AASIST.pth")

    # Initialize model
    model = get_model(config["model_config"], device)

    # Load weights
    if os.path.exists(config["model_path"]):
        state_dict = torch.load(config["model_path"], map_location=device)

        # Handle DataParallel weights if needed
        if list(state_dict.keys())[0].startswith("module."):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print(f"Loaded AASIST weights from {config['model_path']}")
    else:
        print(f"Warning: AASIST weight file {config['model_path']} not found")
        return

    # Create dataset and dataloader with safe collate
    dataset = MP3Dataset(mp3_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Smaller batch size
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate_fn,
    )

    # Run evaluation with details
    try:
        acc, auroc, eer, outputs_list, labels_list, file_paths_list, preds = (
            run_validation_tm_with_details(config, dataloader, model, device)
        )

        # Save results to file
        save_results_to_file(
            results_file,
            "AASIST",
            acc,
            auroc,
            eer,
            outputs_list,
            labels_list,
            file_paths_list,
            preds,
        )

    except Exception as e:
        print(f"Error evaluating AASIST: {e}")


def main():
    # Paths
    mp3_dir = "../samples"  # Directory containing MP3 files
    weights_dir = "./models/weights"  # Directory containing model weights
    results_file = "results.txt"  # Output results file

    if not os.path.exists(mp3_dir):
        print(f"MP3 directory {mp3_dir} not found!")
        return

    if not os.path.exists(weights_dir):
        print(f"Weights directory {weights_dir} not found!")
        return

    # Clear results file
    with open(results_file, "w") as f:
        f.write("Audio Deepfake Detection Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"MP3 files directory: {mp3_dir}\n")
        f.write(f"Model weights directory: {weights_dir}\n")

    print("=== Audio Deepfake Detection Evaluation ===")
    print(f"MP3 files directory: {mp3_dir}")
    print(f"Model weights directory: {weights_dir}")
    print(f"Results will be saved to: {results_file}")

    # Evaluate foundation models
    print("\n--- Foundation Models ---")
    evaluate_foundation_models(mp3_dir, weights_dir, results_file)

    # Evaluate AASIST
    print("\n--- Traditional Model ---")
    evaluate_aasist(mp3_dir, weights_dir, results_file)

    print(f"\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print(f"Detailed results saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
