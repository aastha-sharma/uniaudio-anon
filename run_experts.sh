#!/bin/bash
echo "1. Starting LogMel Training..."
python src/train/train_expert.py \
  --manifest data/cross_domain/manifest_speech_only.csv \
  --kind logmel \
  --epochs 10 \
  --batch 64 \
  --lr 0.001

echo "2. Starting MFCC Training..."
python src/train/train_expert.py \
  --manifest data/cross_domain/manifest_speech_only.csv \
  --kind mfcc \
  --epochs 10 \
  --batch 64 \
  --lr 0.001

echo "3. Starting Wav2Vec Training..."
python src/train/train_expert.py \
  --manifest data/cross_domain/manifest_speech_only.csv \
  --kind w2v_rn \
  --epochs 8 \
  --batch 8 \
  --lr 1e-4

echo " ALL EXPERTS FINISHED!"
