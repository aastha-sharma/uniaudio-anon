import torch
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from unified_training.unified_model_3branch import UnifiedDeepfakeDetector3Branch
    
    ckpt_path = 'checkpoints/unified_3branch_speech_FINAL/best_stage3_3branch.pth'
    print(f"Checking baseline: {ckpt_path}")

    # Load model structure
    model = UnifiedDeepfakeDetector3Branch(checkpoint_dir='./checkpoints')
    
    # Load weights
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
        
    model.load_state_dict(state, strict=False)
    model.eval()
    
    print("\n--- Running Dummy Inference ---")
    # Create fake audio batch (Batch Size 4, 4 seconds of audio)
    dummy_input = torch.randn(4, 64000) 
    
    with torch.no_grad():
        # returns (logit, gate_weights, ...)
        _, gate_weights, _ = model(dummy_input, return_expert_outputs=True)
        
    print("\nGATE PROBABILITIES (Avg over batch):")
    gw = gate_weights.mean(dim=0)
    print(f"  LogMel Expert:  {gw[0]:.6f}")
    print(f"  MFCC Expert:    {gw[1]:.6f}")
    print(f"  Wav2Vec Expert: {gw[2]:.6f}")
    
    if gw.max() > 0.95:
        print("\nDIAGNOSIS: COLLAPSED (Bad)")
        print("The baseline model is already ignoring 2 out of 3 experts.")
    else:
        print("\nDIAGNOSIS: HEALTHY (Good)")
        print("The baseline model is using a mix of experts.")

except Exception as e:
    print(f"\nError: {e}")
    print("Make sure you are running this from the folder containing 'unified_training/'")