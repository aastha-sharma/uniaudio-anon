
UniAudio: A Unified Feature Mixture Framework for Joint Speech and
Singing Deepfake Detection

## Abstract

High-fidelity audio generation techniques, such as voice conversion and singing voice synthesis, have significantly increased the risk of audio deepfakes. Although existing methods perform well on conversational speech deepfake detection, they fail severely under the speech-to-singing domain shift. To address this limitation, we propose **UniAudio**, a unified deepfake detector based on a multi-branch mixture-of-experts architecture that integrates three complementary feature views: Wav2Vec 2.0 representations, log-mel spectrograms, and mel-frequency cepstral coefficients (MFCC). Each expert is trained to remain independently discriminative, while a learned gating network dynamically weights expert contributions. A speech-retentive multi-domain fine-tuning strategy enables adaptation to singing without degrading speech performance. UniAudio reduces singing-domain EER from 43% to 1.82% on CtrSVDD, while preserving strong speech performance (0.38% EER) on ASVspoof2019.



## Table of Contents

1. [Prerequisites and Dependencies](#1-prerequisites-and-dependencies)
2. [Installation](#2-installation)
3. [Data Availability and Preparation](#3-data-availability-and-preparation)
4. [Reproducing Results](#4-reproducing-results)
5. [Code Structure](#5-code-structure)



## 1. Prerequisites and Dependencies

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU, 8GB VRAM | NVIDIA RTX 3090 / A100, 24GB+ VRAM |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 200GB (for all datasets) |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0.0+ | Deep learning framework |
| torchaudio | 2.0.0+ | Audio processing |
| transformers | 4.30.0+ | Wav2Vec 2.0 model |
| CUDA | 11.8+ | GPU acceleration |
| FFmpeg | 4.0+ | Audio conversion |

### Python Dependencies

All dependencies are specified in `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
tqdm>=4.65.0
demucs>=4.0.0
soundfile>=0.12.0
```

---

## 2. Installation

### Step 1: Clone Repository

```bash
git clone https://anonymous.4open.science/r/uniaudio-anon-E3B1/
cd uniaudio-anon
```

### Step 2: Create Environment

```bash
python -m venv uniaudio_env
source uniaudio_env/bin/activate  # Linux/Mac
# or: uniaudio_env\Scripts\activate  # Windows
```

### Step 3: Install PyTorch

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# CPU only (not recommended for training)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import Wav2Vec2Model; print('Transformers OK')"
```

---

## 3. Data Availability and Preparation

### 3.1 Dataset Sources

| Dataset | Domain | Samples | Source | Access |
|---------|--------|---------|--------|--------|
| ASVspoof 2019 LA | Speech | 121,461 | [ASVspoof.org](https://www.asvspoof.org/index2019.html) | Registration required |
| WaveFake | Speech | 196,000+ | [GitHub](https://github.com/RUB-SysSec/WaveFake) | Public |
| CtrSVDD | Singing | 20,765 | [GitHub](https://github.com/SVDDChallenge/CtrSVDD2024_Baseline) | Public |
| SingFake | Singing | 28,000+ | [GitHub](https://github.com/yongyizang/SingFake) | Public |
| ASVspoof 2021 | Speech | 611,829 | [ASVspoof.org](https://www.asvspoof.org/index2021.html) | Registration required |

### 3.2 Directory Structure

After downloading, organize datasets as follows:

```
data/
├── asvspoof2019/
│   ├── LA/
│   │   ├── ASVspoof2019_LA_train/
│   │   ├── ASVspoof2019_LA_dev/
│   │   └── ASVspoof2019_LA_eval/
│   └── LA_cm_protocols/
├── asvspoof2021/
│   └── ASVspoof2021_LA_eval/
├── wavefake/
│   ├── train/
│   ├── val/
│   └── test/
├── ctrsvdd/
│   ├── train_set/
│   └── dev_set/
└── singfake/
    ├── bonafide/
    └── spoof/
```

### 3.3 CtrSVDD Dataset Construction

We organize the CtrSVDD dataset using the official metadata lists (`train.txt`, `dev.txt`). In our pipeline, the official CtrSVDD `dev` partition is mapped to the `test` split for evaluation purposes.

**Note on External Corpora:** Some bonafide segments referenced by the official metadata are absent from the base SVDD Challenge download due to licensing restrictions. To reconstruct the complete dataset:

1. Download the optional external corpora (JVS-Music, Kiritan, Oniku, Ofuton).
2. Use the provided `tools/prepare_ctrsvdd.py` script with the corresponding timestamp files.

The script automatically segments the required time ranges from these external corpora and saves them as 16kHz mono FLAC files into the correct directory structure (`train/bonafide`).

```bash
# Example: Prepare CtrSVDD with external corpora
python tools/prepare_ctrsvdd.py \
    --ctrsvdd_root ./data/ctrsvdd \
    --external_root ./data/external_corpora \
    --output_root ./data/ctrsvdd_complete
```

### 3.4 Create Manifest File

```bash
# Set dataset paths
export ASV2019_ROOT=./data/asvspoof2019/LA
export ASV2019_PROTOCOLS=./data/asvspoof2019/LA_cm_protocols
export ASV2021_ROOT=./data/asvspoof2021
export SINGING_ROOT=./data/ctrsvdd
export SINGFAKE_ROOT=./data/singfake
export WAVEFAKE_ROOT=./data/wavefake
export DERIVED_ROOT=./data/processed

# Generate manifest
python scripts/create_manifest.py \
    --out manifest.csv \
    --derived_root $DERIVED_ROOT \
    --asvspoof2019 $ASV2019_ROOT \
    --asvspoof2019_protocols $ASV2019_PROTOCOLS \
    --asvspoof2021 $ASV2021_ROOT \
    --singing $SINGING_ROOT \
    --singfake $SINGFAKE_ROOT \
    --wavefake $WAVEFAKE_ROOT

# Validate manifest
python tools/validate_manifest.py --manifest manifest.csv
```

### 3.5 Vocal Separation (Singing Datasets)

```bash
# Extract vocals using Demucs (GPU recommended)
python scripts/extract_vocals.py --manifest manifest.csv --device cuda
```

---

## 4. Reproducing Results

This section provides commands to reproduce all experimental results reported in the paper.

### 4.1 Model Checkpoints

**Note:** Pre-trained model checkpoints are not included in this repository due to file size constraints. All checkpoints can be reproduced by running the training scripts below.

After training, checkpoints will be saved to the `checkpoints/` directory:
```
checkpoints/
├── logmel_pretrained.pth
├── mfcc_pretrained.pth
├── w2v_rn_pretrained.pth
└── unified_3branch_FINAL/
    └── best_stage3_3branch.pth
```

### 4.2 Table 2: Zero-Shot Evaluation

**Purpose:** Demonstrate speech-to-singing domain shift failure.

```bash
# Train speech-only baseline (Stages 1-2)
python unified_training/train_unified_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/speech_only_baseline \
    --stage1-epochs 10 \
    --stage2-epochs 25 \
    --stage3-epochs 0

# Evaluate on all test splits
python evaluate_metrics.py \
    --model checkpoints/speech_only_baseline/best_stage2_3branch.pth \
    --manifest manifest.csv \
    --split test \
    --output results/table2_zero_shot.json
```

### 4.3 Table 3: Expert Branch Analysis

**Purpose:** Compare individual expert generalization across domains.

```bash
# Train individual experts
for EXPERT in logmel mfcc w2v_rn; do
    python src/train/train_expert.py \
        --manifest manifest.csv \
        --kind $EXPERT \
        --epochs 10 \
        --batch 64 \
        --lr 0.001
done

# Evaluate each expert
for EXPERT in logmel mfcc w2v_rn; do
    python evaluate_metrics.py \
        --model checkpoints/${EXPERT}_pretrained.pth \
        --manifest manifest.csv \
        --split test \
        --output results/table3_${EXPERT}_expert.json
done
```

### 4.4 Table 4: Multi-Domain Fine-Tuning (Main Result)

**Purpose:** Reproduce main UniAudio results with speech-retentive fine-tuning.

```bash
# Full three-stage training with 50/50 multi-domain fine-tuning
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --base-model checkpoints/speech_only_baseline/best_stage2_3branch.pth \
    --output-dir checkpoints/unified_multidomain \
    --stage3-epochs 5 \
    --batch-size 16 \
    --lr 3e-5 \
    --speech-ratio-s3 0.5 \
    --lambda-domain 0.0 \
    --lambda-aux 0.1 \
    --lambda-ent 1e-4 \
    --seed 42

# Evaluate final model
python evaluate_metrics.py \
    --model checkpoints/unified_multidomain/best_stage3_fewshot.pth \
    --manifest manifest.csv \
    --split test \
    --output results/table4_main_results.json
```

### 4.5 Table 5: DANN Baseline Comparison

**Purpose:** Compare against adversarial domain alignment.

```bash
# Configuration 1: 10% singing, Stage 3
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/dann_cfg1 \
    --speech-ratio-s3 0.9 \
    --lambda-domain 0.3 \
    --stage3-epochs 15

# Configuration 2: 30% singing, Stage 3
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/dann_cfg2 \
    --speech-ratio-s3 0.7 \
    --lambda-domain 0.3 \
    --stage3-epochs 15

# Pre-computed results available in results/table5_adaptation_sweep/
```

### 4.6 Tables 6-10: Ablation Studies

**Table 6: Loss Component Analysis**
```bash
# Baseline (L_task only)
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/ablation_ltask_only \
    --lambda-aux 0.0 --lambda-ent 0.0 --lambda-div 0.0

# +Auxiliary loss
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/ablation_plus_aux \
    --lambda-aux 0.1 --lambda-ent 0.0 --lambda-div 0.0

# +Entropy regularization
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/ablation_plus_ent \
    --lambda-aux 0.1 --lambda-ent 1e-4 --lambda-div 0.0

# Full model (all components)
python unified_training/train_dann_3branch.py \
    --manifest manifest.csv \
    --output-dir checkpoints/ablation_full \
    --lambda-aux 0.1 --lambda-ent 1e-4 --lambda-div 0.1
```

**Table 7: Entropy Regularization Sweep (λ_ent)**
```bash
for ENT in 0 1e-5 1e-4 5e-4 1e-3; do
    python unified_training/train_dann_3branch.py \
        --manifest manifest.csv \
        --output-dir checkpoints/sweep_ent_${ENT} \
        --lambda-ent ${ENT} \
        --speech-ratio-s3 0.5
done
```

**Table 8: Diversity Penalty Sweep (λ_div)**
```bash
for DIV in 0 0.05 0.1 0.2 0.5; do
    python unified_training/train_dann_3branch.py \
        --manifest manifest.csv \
        --output-dir checkpoints/sweep_div_${DIV} \
        --lambda-div ${DIV}
done
```

**Table 9: Temperature Annealing Schedule**

Temperature annealing is configured in `unified_training/train_unified_3branch.py`. Pre-computed results for different schedules are available in `results/table9_temp_schedule_sweep/`.

**Table 10: Speech-Singing Mixing Ratio**
```bash
for RATIO in 0.9 0.7 0.5 0.3 0.1; do
    python unified_training/train_dann_3branch.py \
        --manifest manifest.csv \
        --output-dir checkpoints/sweep_ratio_${RATIO} \
        --speech-ratio-s3 ${RATIO}
done
```



## 5. Code Structure

```
uniaudio-anon/
│
├── src/                              # Core source code
│   ├── data/
│   │   └── dataset.py                # PyTorch Dataset classes
│   ├── features/
│   │   ├── extractors.py             # Log-Mel, MFCC feature extraction
│   │   └── wav2vec.py                # Wav2Vec 2.0 wrapper
│   ├── models/
│   │   ├── experts_logmel_mfcc.py    # Spectral/Cepstral expert networks
│   │   └── experts_w2v.py            # Wav2Vec 2.0 expert network
│   └── train/
│       └── train_expert.py           # Individual expert training
│
├── unified_training/                 # Unified model training
│   ├── unified_model_3branch.py      # Main UniAudio architecture (Fig. 1)
│   ├── train_unified_3branch.py      # Three-stage training pipeline
│   ├── train_dann_3branch.py         # DANN baseline + multi-domain fine-tuning
│   ├── gating_network.py             # Adaptive gating implementation
│   └── cross_domain_dataset.py       # Domain-balanced data loading
│
├── scripts/                          # Data preprocessing
│   ├── create_manifest.py            # Generate dataset manifest
│   ├── extract_vocals.py             # Demucs vocal separation
│   └── preprocess_audio.py           # Audio normalization
│
├── tools/                            # Utilities
│   ├── validate_manifest.py          # Manifest validation
│   ├── prepare_ctrsvdd.py            # CtrSVDD dataset preparation
│   └── download_singfake.py          # SingFake download helper
│
├── results/                          # Pre-computed experimental results
│
├── evaluate_metrics.py               # Evaluation script (EER, AUC)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### Key File Descriptions

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `unified_model_3branch.py` | UniAudio architecture | Section 3, Figure 1 |
| `train_unified_3branch.py` | Three-stage training | Section 3.4, Section 4.3 |
| `gating_network.py` | Adaptive gating with entropy regularization | Section 3.3, Eq. 1-3 |
| `experts_*.py` | Expert branch implementations | Section 3.2 |
| `evaluate_metrics.py` | EER/AUC computation | Section 5 |


