# Multimodal FER + EEG Deception Detection

A PyTorch-based multimodal verification system combining Facial Emotion Recognition (FER) and EEG Brainwave Analysis for cross-modal deception detection.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐
│   Face Image    │     │   EEG Signal     │
│   (B, 3, 224,224)│     │  (B, 32, 1024)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  Visual Stream  │     │   EEG Stream     │
│  (ResNet + CBAM)│     │ (1D-CNN + BiLSTM)│
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │   z_face (B, 256)     │   z_eeg (B, 256)
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌────────────────────┐
          │  Fusion Module     │
          │ (Cross-Attention)  │
          └────────┬───────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ Fused Classifier │ │ Deception Head   │
│ (Emotion 8-class)│ │ (Genuine/Decept) │
└──────────────────┘ └──────────────────┘
```

## Features

- **Visual Stream**: ResNet18/34/50 backbone with CBAM attention for FER
- **EEG Stream**: 1D-CNN + BiLSTM processing temporal + spectral (PSD) features
- **Fusion**: Cross-attention, concatenation, or bilinear fusion
- **Deception Detection**: Similarity-based or MLP classifier for incongruence detection
- **Multi-task Loss**: αL_face + βL_eeg + γL_fusion + λL_congruence
- **Mixed Precision**: Automatic mixed precision (AMP) support
- **TensorBoard Logging**: Full training metrics and confusion matrices
- **Subject-wise Splits**: Prevents data leakage across train/val/test

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The dataset should be organized as:
```
archive/
├── emotions.csv          # Metadata: set_id, gender, age, country
└── images/
    ├── 0/
    │   ├── Anger.jpg
    │   ├── Contempt.jpg
    │   ├── Disgust.jpg
    │   ├── Fear.jpg
    │   ├── Happy.jpg
    │   ├── Neutral.jpg
    │   ├── Sad.jpg
    │   └── Surprised.jpg
    ├── 1/
    │   └── ...
    └── 18/
        └── ...
```

**Note**: The current dataset only contains facial images. EEG data is synthetically generated based on established EEG-emotion correlates. When real EEG data is available, place `.npy` files in the `eeg_root` directory and update the config.

## Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (backbone, attention, fusion type)
- Training hyperparameters (lr, batch_size, epochs, loss weights)
- Data paths and preprocessing
- Deception detection threshold

## Usage

### Training

```bash
python main.py --config configs/default.yaml --mode train
```

### Testing/Evaluation

```bash
python main.py --config configs/default.yaml --mode test --checkpoint checkpoints/best_model.pt
```

### Verification (Single Sample)

```bash
python main.py --config configs/default.yaml --mode verify \
    --checkpoint checkpoints/best_model.pt \
    --face-image path/to/face.jpg \
    --eeg-file path/to/eeg.npy  # Optional
```

Output:
```
=== Verification Result ===
  predicted_face_emotion: Happy
  predicted_eeg_emotion: Happy
  fused_emotion: Happy
  authenticity_score: 0.92
  is_deceptive: False
  deception_probability: 0.08
  similarity_score: 0.87
  valence: 0.85
  arousal: 0.68

=== Diagnostics ===
  similarity_score: 0.87
  deception_probability: 0.08
  valence: 0.85
  arousal: 0.68
  face_emotion: Happy
  eeg_emotion: Happy
  fused_emotion: Happy
  valence_interpretation: positive
  arousal_interpretation: high
  incongruence_explanation: Face and EEG are congruent. Face: 'Happy', EEG: 'Happy'. Similarity=0.870 >= threshold=0.7
```

## Model Components

### Visual Stream (`models/models.py`)
- Backbone: ResNet18/34/50 or MobileNetV3
- Attention: CBAM (Channel + Spatial), Spatial-only, or None
- Output: 256-dim latent + 8-class emotion logits

### EEG Stream (`models/models.py`)
- 1D-CNN: 3 layers (64, 128, 256 channels) with max-pooling
- BiLSTM: 2 layers, 256 hidden, bidirectional
- Spectral branch: PSD features (Theta, Alpha, Beta, Gamma) per channel
- Output: 256-dim latent + 8-class logits + valence/arousal regression

### Fusion (`models/models.py`)
- **Cross-Attention**: Bidirectional attention between modalities
- **Concat**: MLP on concatenated features
- **Bilinear**: Hadamard product with low-rank projection

### Deception Head (`models/models.py`)
- **similarity_binary**: Learnable mapping from cosine similarity → deception prob
- **mlp_classifier**: MLP on concatenated [z_face, z_eeg]
- **distance_threshold**: Fixed threshold on similarity/distance

### Loss Functions (`loss/loss.py`)
- **FER Loss**: Cross-entropy with label smoothing
- **EEG Loss**: Cross-entropy + MSE (valence) + MSE (arousal)
- **Fusion Loss**: Cross-entropy
- **Congruence Loss**: BCE / Contrastive / Triplet / Similarity regression

## Real EEG Data Integration

When real EEG data is available:

1. Place preprocessed EEG files (`.npy`, shape: C×T) in `eeg_root/`
2. Update `configs/default.yaml`:
   ```yaml
   data:
     eeg_root: "path/to/eeg_data"
   ```
3. Implement `_load_real_eeg()` in `dataset/dataset.py` to match your file naming convention
4. The dataset will automatically use real EEG when available

Expected EEG format per sample:
- Temporal: `(n_channels, n_samples)` e.g., `(32, 1024)` for 4s @ 256Hz
- Spectral: Computed automatically via PSD (Theta/Alpha/Beta/Gamma bands)

## Project Structure

```
multimodal_fer_eeg/
├── configs/
│   └── default.yaml           # Main configuration
├── dataset/
│   └── dataset.py             # Data loading, preprocessing, synthetic EEG
├── models/
│   └── models.py              # Model architectures
├── loss/
│   └── loss.py                # Multi-task loss functions
├── engine/
│   └── engine.py              # Training loop, validation, checkpointing
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## Citation

If you use this code for research, please cite the relevant papers for:
- ResNet (He et al., 2016)
- CBAM (Woo et al., 2018)
- EEG emotion recognition literature
- Cross-modal deception detection literature

## License

MIT License