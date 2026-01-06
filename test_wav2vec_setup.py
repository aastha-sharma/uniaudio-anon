# test_wav2vec_setup.py
"""
Quick test to verify Wav2Vec2 components are ready
"""
import torch
import sys

print("="*70)
print("TESTING WAV2VEC2 SETUP")
print("="*70)

# Test 1: Check if transformers library available
print("\n1. Checking transformers library...")
try:
    from transformers import Wav2Vec2Model
    print("   ✅ transformers library installed")
except ImportError as e:
    print(f"   ❌ transformers not found: {e}")
    print("   Install with: pip install transformers")
    sys.exit(1)

# Test 2: Check if wav2vec extractor exists
print("\n2. Checking Wav2Vec2 feature extractor...")
try:
    from src.features.wav2vec import Wav2VecExtractor
    print("   ✅ Wav2VecExtractor found in src.features.wav2vec")
except ImportError as e:
    print(f"   ❌ Wav2VecExtractor not found: {e}")
    print("   Need to create src/features/wav2vec.py")
    sys.exit(1)

# Test 3: Check if w2v models exist
print("\n3. Checking Wav2Vec2 model implementations...")
try:
    from src.models.experts_w2v import Wav2VecTransformerExpert, Wav2VecResNetExpert
    print("   ✅ Wav2Vec2 expert models found")
except ImportError as e:
    print(f"   ❌ Wav2Vec2 models not found: {e}")
    print("   Need to create src/models/experts_w2v.py")
    sys.exit(1)

# Test 4: Check if train_expert supports w2v
print("\n4. Checking training script support...")
try:
    from src.train.train_expert import make_model
    
    # Try to create w2v models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n   Testing Wav2Vec2-Transformer model creation...")
    try:
        model_tx = make_model('w2v_transformer')
        print(f"   ✅ Wav2Vec2-Transformer created: {sum(p.numel() for p in model_tx.parameters()):,} params")
        del model_tx
    except Exception as e:
        print(f"   ❌ Failed to create w2v_transformer: {e}")
    
    print("\n   Testing Wav2Vec2-ResNet model creation...")
    try:
        model_rn = make_model('w2v_resnet')
        print(f"   ✅ Wav2Vec2-ResNet created: {sum(p.numel() for p in model_rn.parameters()):,} params")
        del model_rn
    except Exception as e:
        print(f"   ❌ Failed to create w2v_resnet: {e}")
        
except Exception as e:
    print(f"   ❌ Error checking training support: {e}")
    sys.exit(1)

# Test 5: Quick forward pass
print("\n5. Testing forward pass...")
try:
    model = make_model('w2v_transformer').to(device)
    dummy_audio = torch.randn(2, 96000).to(device)  # 2 samples, 6 seconds
    
    with torch.no_grad():
        output = model(dummy_audio)
    
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape: {dummy_audio.shape}")
    print(f"   Output shape: {output.shape}")
    
    del model, dummy_audio, output
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED! Ready to train Wav2Vec2 branches")
print("="*70)
print("\nNext steps:")
print("1. Train Wav2Vec2-Transformer:")
print("   python -m src.train.train_expert --manifest manifest.csv --kind w2v_transformer --epochs 6 --batch 16 --lr 3e-4")
print("\n2. Train Wav2Vec2-ResNet:")
print("   python -m src.train.train_expert --manifest manifest.csv --kind w2v_resnet --epochs 6 --batch 16 --lr 3e-4")
