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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F

# Import existing modules
from main_fm import run_validation as run_validation_fm
from models import Hubert, Wav2Vec2BERT, Wav2Vec2
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


def calculate_predictions(
    outputs_list, labels_list, eer, threshold_type, fixed_threshold=0.5
):
    """Calculate predictions based on threshold type"""

    if threshold_type == "eer":
        threshold = eer[1]
        threshold_name = f"EER ({threshold:.4f})"
    elif threshold_type == "fixed":
        threshold = fixed_threshold
        threshold_name = f"Fixed ({threshold:.4f})"
    else:
        raise ValueError(
            f"Invalid threshold_type: {threshold_type}. Use 'eer' or 'fixed'"
        )

    preds = (np.array(outputs_list) > threshold).astype(int)
    acc = np.mean(np.array(labels_list) == preds)

    print(f"Using {threshold_name} threshold: Accuracy = {acc:.4f}")

    return preds, acc, threshold


def run_validation_fm_with_details(
    model,
    feature_extractor,
    data_loader,
    sr,
    threshold_type="fixed",
    fixed_threshold=0.5,
):
    """Modified version that returns individual predictions with configurable threshold"""
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
            )  # probability for class 1 (genuine)
            labels_list.extend(batch_label)
            file_paths_list.extend(names)

    # Calculate metrics
    auroc = roc_auc_score(labels_list, outputs_list)
    eer = compute_eer(np.array(labels_list), np.array(outputs_list))

    # Debug info
    print(f"\n=== DEBUG FM ===")
    print(f"EER: {eer[0]:.4f}, EER Threshold: {eer[1]:.4f}")
    print(f"Prob range: {min(outputs_list):.4f} to {max(outputs_list):.4f}")

    # Calculate predictions based on threshold type
    preds, acc, used_threshold = calculate_predictions(
        outputs_list, labels_list, eer, threshold_type, fixed_threshold
    )

    return (
        acc,
        auroc,
        eer,
        outputs_list,
        labels_list,
        file_paths_list,
        preds,
        used_threshold,
    )


def run_validation_tm_with_details(
    config, data_loader, model, device, threshold_type="fixed", fixed_threshold=0.5
):
    """Modified version that returns individual predictions with configurable threshold"""
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
            )  # probability for class 1 (genuine)
            labels_list.extend(batch_label)
            file_paths_list.extend(names)

    # Calculate metrics
    auroc = roc_auc_score(labels_list, outputs_list)
    eer = compute_eer(np.array(labels_list), np.array(outputs_list))

    # Debug info
    print(f"\n=== DEBUG TM ===")
    print(f"Model: {config['model_config']['architecture']}")
    print(f"EER: {eer[0]:.4f}, EER Threshold: {eer[1]:.4f}")
    print(f"Prob range: {min(outputs_list):.4f} to {max(outputs_list):.4f}")

    # Calculate predictions based on threshold type
    preds, acc, used_threshold = calculate_predictions(
        outputs_list, labels_list, eer, threshold_type, fixed_threshold
    )

    return (
        acc,
        auroc,
        eer,
        outputs_list,
        labels_list,
        file_paths_list,
        preds,
        used_threshold,
    )


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
    used_threshold,
    threshold_type,
):
    """Save detailed results to file"""
    with open(results_file, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"MODEL: {model_name.upper()}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {acc:.4f}\n")
        f.write(f"  AUROC: {auroc:.4f}\n")
        f.write(f"  EER: {eer[0]:.4f}\n")
        f.write(f"  EER Threshold: {eer[1]:.4f}\n")
        f.write(f"  Used Threshold: {used_threshold:.4f} ({threshold_type})\n\n")

        f.write("Individual File Results:\n")
        f.write("Filename\tTrue_Label\tPredicted\tProb_Genuine\tConfidence\tCorrect\n")
        f.write("-" * 80 + "\n")

        for i, (file_path, true_label, pred, prob_genuine) in enumerate(
            zip(file_paths_list, labels_list, preds, outputs_list)
        ):
            filename = os.path.basename(file_path)
            correct = "yes" if true_label == pred else "no"

            # Calculate confidence: probability of the predicted class
            confidence = prob_genuine if pred == 1 else (1.0 - prob_genuine)

            f.write(
                f"{filename}\t{true_label}\t{pred}\t{prob_genuine:.4f}\t{confidence:.4f}\t{correct}\n"
            )
        f.write("\n")


def evaluate_foundation_models(
    mp3_dir, weights_dir, results_file, threshold_type, fixed_threshold
):
    """Evaluate HuBert and Wav2Vec2BERT using main_fm functions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models to evaluate
    models_config = {
        # "hubert": {
        #     "model_name": "facebook/hubert-large-ls960-ft",
        #     "weight_file": "hubert_large_wavefake.pth",
        #     "sampling_rate": 16000,
        # },
        # "wav2vec2": {  # Changed from "hubert"
        #     "model_name": "facebook/wav2vec2-large-960h",  # Matches the checkpoint architecture
        #     "weight_file": "hubert_large_wavefake.pth",  # Your existing checkpoint
        #     "sampling_rate": 16000,
        # },
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

    result_blocks = []

    for model_name, config in models_config.items():
        print(f"\nEvaluating {model_name}...")

        # Initialize model
        if model_name == "hubert":
            model = Hubert(config["model_name"])
        elif model_name == "wav2vec2":  # THIS WAS MISSING
            model = Wav2Vec2(config["model_name"])
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
            (
                acc,
                auroc,
                eer,
                outputs_list,
                labels_list,
                file_paths_list,
                preds,
                used_threshold,
            ) = run_validation_fm_with_details(
                model,
                feature_extractor,
                dataloader,
                config["sampling_rate"],
                threshold_type,
                fixed_threshold,
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
                used_threshold,
                threshold_type,
            )

            block = _build_block(
                model_name,
                threshold_type,
                eer,
                used_threshold,
                file_paths_list,
                preds,
                outputs_list,
                labels_list,
            )  # ← add
            result_blocks.append(block)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

        return result_blocks


def evaluate_traditional_models(
    mp3_dir, weights_dir, results_file, threshold_type, fixed_threshold
):
    """Evaluate traditional models: AASIST, RawGATST, RawNet2"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model configurations with manually specified weight paths
    models_config = {
        "AASIST": {
            "config_path": "./config/AASIST.conf",
            "weight_path": os.path.join(
                weights_dir, "AASIST.pth"
            ),  # Adjust path as needed
        },
        # "RawGATST": {
        #     "config_path": "./config/RawGATST.conf",
        #     "weight_path": "./models/weights/rawgatst.pth",  # **UPDATE THIS PATH**
        # },
        "RawNet2": {
            "config_path": "./config/RawNet2.conf",
            "weight_path": "./models/weights/rawnet2.pth",  # **UPDATE THIS PATH**
        },
    }

    # Create dataset and dataloader with safe collate
    dataset = MP3Dataset(mp3_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Smaller batch size
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate_fn,
    )

    result_blocks = []

    for model_name, model_info in models_config.items():
        print(f"\nEvaluating {model_name}...")

        # Check if weight file exists
        if not os.path.exists(model_info["weight_path"]):
            print(
                f"Warning: Weight file {model_info['weight_path']} not found, skipping {model_name}"
            )
            continue

        # Load config
        if not os.path.exists(model_info["config_path"]):
            print(
                f"Warning: Config file {model_info['config_path']} not found, skipping {model_name}"
            )
            continue

        with open(model_info["config_path"], "r") as f:
            config = json.loads(f.read())

        # Initialize model
        try:
            model = get_model(config["model_config"], device)
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
            continue

        # Load weights
        try:
            state_dict = torch.load(model_info["weight_path"], map_location=device)

            # Handle DataParallel weights if needed
            if list(state_dict.keys())[0].startswith("module."):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            model.load_state_dict(state_dict)
            print(f"Loaded {model_name} weights from {model_info['weight_path']}")
        except Exception as e:
            print(f"Error loading weights for {model_name}: {e}")
            continue

        # Run evaluation with details
        try:
            (
                acc,
                auroc,
                eer,
                outputs_list,
                labels_list,
                file_paths_list,
                preds,
                used_threshold,
            ) = run_validation_tm_with_details(
                config, dataloader, model, device, threshold_type, fixed_threshold
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
                used_threshold,
                threshold_type,
            )

            block = _build_block(
                model_name,
                threshold_type,
                eer,
                used_threshold,
                file_paths_list,
                preds,
                outputs_list,
                labels_list,
            )  # ← add
            result_blocks.append(block)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    return result_blocks


# ── Helper to convert raw per-file outputs into a plotting block ───────────────
from collections import defaultdict
import os, numpy as np

_SPK = ("lw", "lhl", "jt", "gky")  # speakers we care about


def _build_block(title, threshold_type, eer, thr, file_paths, preds, probs, labels):
    """
    Collapse per-file results into the minimal structure needed by
    plot_result_tables(), now with AUROC & F1 in the metadata.
    """
    # --- Compute overall AUROC and F1 for this model -------------------------
    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)  # ← compute accuracy

    data = {sid: defaultdict(list) for sid in _SPK}
    correct = {sid: defaultdict(list) for sid in _SPK}

    for fp, p, pr, lab in zip(file_paths, preds, probs, labels):
        fname = os.path.basename(fp)
        sid, m = fname.split("_", 2)[:2]  #  "lw_OpenAI_xxx.mp3" → "lw", "OpenAI"
        if sid not in _SPK:
            continue
        conf = pr if p == 1 else (1.0 - pr)  # confidence of predicted class
        data[sid][m].append(conf)
        correct[sid][m].append(int(p == lab))

    # aggregate: mean confidence + “all correct?”
    block = {
        "title": title,
        "threshold_type": threshold_type,
        "eer": float(eer[0]),
        "threshold": float(thr),
        "auroc": float(auroc),  # ← added
        "f1": float(f1),  # ← added
        "acc": float(acc),
        "data": {
            sid: {
                m: (float(np.mean(vals)), bool(all(correct[sid][m])))
                for m, vals in methods.items()
            }
            for sid, methods in data.items()
            if methods
        },
    }
    return block


# ──────────────────────────────────────────────────────────────────────────────
# 📊  Minimal table-plotting helper
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# Four speakers we care about → pretty display names
_DISPLAY_NAME = {
    "lw": "Lawrence Wong",
    "lhl": "Lee Hsien Loong",
    "jt": "Josephine Teo",
    "gky": "Gan Kim Yong",
}

# Canonical column order + pretty labels
_METHOD_ORDER = ["original", "oaii", "oaip", "el", "ov", "cq"]
_METHOD_LABELS = {
    "original": "Original",
    "oaii": "OpenAI Instance",
    "oaip": "OpenAI Pro",
    "el": "ElevenLabs",
    "ov": "OpenVoice",
    "cq": "Coqui (Bark)",
}


def _canon(m: str) -> str:
    """Map raw method strings (sometimes messy) to canonical short codes."""
    m = m.lower().replace(".mp3", "")  # strip extension just in case
    if "openvoice" in m or m == "ov":
        return "ov"
    if "openai" in m and "pro" in m:
        return "oaip"
    if "openai" in m:
        return "oaii"
    if "eleven" in m or m == "el":
        return "el"
    if "coqui" in m or "bark" in m or m == "cq":
        return "cq"
    if "orig" in m or "genuine" in m:
        return "original"
    return m  # fall-back: use as-is


def plot_result_tables(result_blocks, outfile: str = "tables.png") -> None:
    """
    Draw one coloured table per *result_blocks* element (see _build_block()).

    Figure title  : “Results and Confidence Scores”
    Row order     : Lawrence Wong, Gan Kim Yong, Lee Hsien Loong, Josephine Teo
    Heading lines :   {model_name} – {threshold_type}
                      Thr={thr:.3f}, EER={eer:.3f}, AUROC={auroc:.3f}, F1@Thr={f1:.3f}
    """
    import matplotlib.pyplot as plt

    # ── 1. CONFIG ────────────────────────────────────────────────────────────
    speakers = ("lw", "gky", "lhl", "jt")  # ← reordered to match screenshot

    # ── 2. CREATE FIGURE ─────────────────────────────────────────────────────
    n = len(result_blocks)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2 + 2 * n))
    if n == 1:
        axs = (axs,)  # make iterable for single-row case

    # ── 3. ONE TABLE PER RESULT BLOCK ───────────────────────────────────────
    for ax, blk in zip(axs, result_blocks):
        # ---------- build cell text & colours --------------------------------
        header = ["Name"] + [_METHOD_LABELS[c] for c in _METHOD_ORDER]
        cell_txt, cell_col = [], []

        for sid in speakers:
            row_txt = [_DISPLAY_NAME[sid]]
            row_col = ["white"]  # first cell (names) is white

            for code in _METHOD_ORDER:
                entry = next(
                    (
                        v
                        for k, v in blk["data"].get(sid, {}).items()
                        if _canon(k) == code
                    ),
                    None,
                )
                if entry is None:  # missing entry → grey cell
                    row_txt.append("")
                    row_col.append("lightgrey")
                else:  # entry present
                    score, correct = entry
                    row_txt.append(f"{score * 100:.2f}%")
                    row_col.append("palegreen" if correct else "lightcoral")

            cell_txt.append(row_txt)
            cell_col.append(row_col)

        # ---------- render the table -----------------------------------------
        tbl = ax.table(
            cellText=cell_txt,
            colLabels=header,
            cellColours=cell_col,
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        ax.axis("off")

        # ---------- two-line subplot title -----------------------------------
        title_line1 = (
            f"{blk['title']} – "
            f"{'fixed_threshold' if blk['threshold_type'] == 'fixed' else 'eer_threshold'}"
        )
        title_line2 = (
            f"Threshold={blk['threshold']:.3f}, "
            f"EER={blk['eer']:.3f}, "
            f"Accuracy={blk.get('acc', 0):.3f}, "
            f"AUROC={blk.get('auroc', 0):.3f}, "
            f"F1@Threshold={blk.get('f1', 0):.3f}"
        )
        ax.set_title(f"{title_line1}\n{title_line2}", pad=3, fontsize=12)

    # ── 4. SAVE FIGURE ───────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
    plt.savefig(outfile, dpi=300)
    print(f"[plot_result_tables] saved coloured tables → {outfile}")


def main():
    # ============ CONFIGURATION ============
    # Threshold settings - MODIFY THESE AS NEEDED:

    threshold_type = "eer"  # Options: "eer" or "fixed"
    fixed_threshold = 0.5  # only Used when threshold_type="fixed"

    # =======================================

    # Paths
    mp3_dir = "../samples"  # Directory containing MP3 files
    weights_dir = "./models/weights"  # Directory containing model weights
    results_file = "results_eer.txt"  # Output results file
    table_file = "tables_eer.png"

    if not os.path.exists(mp3_dir):
        print(f"MP3 directory {mp3_dir} not found!")
        return

    # Clear results file
    with open(results_file, "w") as f:
        f.write("Audio Deepfake Detection Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"MP3 files directory: {mp3_dir}\n")
        f.write(f"Model weights directory: {weights_dir}\n")
        f.write(f"Threshold type: {threshold_type}\n")
        if threshold_type == "fixed":
            f.write(f"Fixed threshold: {fixed_threshold}\n")
        f.write("\n")

    print("=== Audio Deepfake Detection Evaluation ===")
    print(f"MP3 files directory: {mp3_dir}")
    print(f"Model weights directory: {weights_dir}")
    print(f"Threshold type: {threshold_type}")
    if threshold_type == "fixed":
        print(f"Fixed threshold: {fixed_threshold}")
    print(f"Results will be saved to: {results_file}")

    # Evaluate foundation models
    print("\n--- Foundation Models ---")
    fm_blocks = evaluate_foundation_models(
        mp3_dir, weights_dir, results_file, threshold_type, fixed_threshold
    )

    # Evaluate traditional models (AASIST, RawGATST, RawNet2)
    print("\n--- Traditional Models ---")
    tm_blocks = evaluate_traditional_models(
        mp3_dir, weights_dir, results_file, threshold_type, fixed_threshold
    )

    print(f"\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print(f"Detailed results saved to: {results_file}")
    print("=" * 50)

    plot_result_tables(
        fm_blocks + tm_blocks, outfile=table_file
    )  # 👈 single call – that's it!


if __name__ == "__main__":
    main()
