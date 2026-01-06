"""
Analyze Cross-Domain Baseline Results

Usage:
    python analyze_baseline_results.py \
        --baseline-singing ./results/baseline_singing.json \
        --baseline-speech ./results/baseline_speech.json
"""
import argparse
import json
from pathlib import Path


def interpret_eer(eer):
    """Interpret EER performance"""
    if eer < 0.05:
        return "Exceptional"
    elif eer < 0.10:
        return "Excellent"
    elif eer < 0.20:
        return "Good"
    elif eer < 0.30:
        return "Fair"
    elif eer < 0.40:
        return "Poor"
    elif eer < 0.50:
        return "Very Poor"
    else:
        return "Failed"


def estimate_dann_improvement(baseline_eer):
    """Estimate potential improvement from DANN"""
    if baseline_eer >= 0.45:
        improvement_low = baseline_eer - 0.30
        improvement_high = baseline_eer - 0.25
        confidence = "High"
        reason = "Large domain gap provides significant room for adaptation"
    elif baseline_eer >= 0.35:
        improvement_low = baseline_eer - 0.25
        improvement_high = baseline_eer - 0.20
        confidence = "Medium-High"
        reason = "Moderate domain gap, DANN can bridge effectively"
    elif baseline_eer >= 0.25:
        improvement_low = baseline_eer - 0.20
        improvement_high = baseline_eer - 0.15
        confidence = "Medium"
        reason = "Model already transfers reasonably, DANN gives polish"
    else:
        improvement_low = baseline_eer - 0.15
        improvement_high = baseline_eer - 0.10
        confidence = "Low"
        reason = "Model already transfers well, limited improvement possible"
    
    return {
        'expected_eer_low': max(0.10, improvement_high),
        'expected_eer_high': improvement_low,
        'improvement_low': improvement_low,
        'improvement_high': improvement_high,
        'confidence': confidence,
        'reason': reason
    }


def analyze_cross_domain_gap(speech_eer, singing_eer):
    """Analyze the domain gap"""
    gap = singing_eer - speech_eer
    
    if gap < 0.05:
        severity = "Minimal"
        recommendation = "DANN may not be necessary"
    elif gap < 0.15:
        severity = "Small"
        recommendation = "DANN could provide incremental improvement"
    elif gap < 0.25:
        severity = "Moderate"
        recommendation = "DANN is recommended"
    elif gap < 0.35:
        severity = "Large"
        recommendation = "DANN is strongly recommended"
    else:
        severity = "Severe"
        recommendation = "DANN is essential"
    
    return {
        'gap': gap,
        'severity': severity,
        'recommendation': recommendation
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-singing", required=True)
    parser.add_argument("--baseline-speech", required=False)
    parser.add_argument("--output", default="./results/baseline_analysis.json")
    
    args = parser.parse_args()
    
    print("="*80)
    print(" "*20 + "CROSS-DOMAIN BASELINE ANALYSIS")
    print("="*80)
    
    # Load singing results
    with open(args.baseline_singing) as f:
        singing_results = json.load(f)
    
    singing_eer = singing_results['overall']['eer']
    
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE ON SINGING (Cross-Domain)")
    print("="*80)
    
    print(f"\nOverall EER: {singing_eer:.4f} ({singing_eer*100:.2f}%)")
    print(f"AUC: {singing_results['overall']['auc']:.4f}")
    print(f"Samples: {singing_results['overall']['num_samples']:,}")
    
    rating = interpret_eer(singing_eer)
    print(f"\n{rating}")
    
    # Per-dataset
    if 'per_dataset' in singing_results and singing_results['per_dataset']:
        print("\n" + "-"*80)
        print("Per-Dataset Performance:")
        for dataset, metrics in singing_results['per_dataset'].items():
            print(f"\n{dataset}: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    
    # Compare with speech
    speech_eer = None
    if args.baseline_speech:
        print("\n" + "="*80)
        print("BASELINE PERFORMANCE ON SPEECH (In-Domain)")
        print("="*80)
        
        with open(args.baseline_speech) as f:
            speech_results = json.load(f)
        
        speech_eer = speech_results['overall']['eer']
        print(f"\nOverall EER: {speech_eer:.4f} ({speech_eer*100:.2f}%)")
        
        gap_analysis = analyze_cross_domain_gap(speech_eer, singing_eer)
        
        print("\n" + "="*80)
        print("DOMAIN GAP ANALYSIS")
        print("="*80)
        print(f"\nSpeech EER: {speech_eer*100:.2f}%")
        print(f"Singing EER: {singing_eer*100:.2f}%")
        print(f"Domain Gap: {gap_analysis['gap']*100:.2f} points")
        print(f"\nSeverity: {gap_analysis['severity']}")
        print(f"Recommendation: {gap_analysis['recommendation']}")
    
    # DANN Recommendation
    print("\n" + "="*80)
    print("DOMAIN ADAPTATION RECOMMENDATION")
    print("="*80)
    
    dann = estimate_dann_improvement(singing_eer)
    
    print(f"\nCurrent EER: {singing_eer*100:.2f}%")
    print(f"Expected with DANN: {dann['expected_eer_low']*100:.2f}%-{dann['expected_eer_high']*100:.2f}%")
    print(f"Expected improvement: {dann['improvement_low']*100:.2f}-{dann['improvement_high']*100:.2f} points")
    print(f"Confidence: {dann['confidence']}")
    print(f"Reason: {dann['reason']}")
    
    # Decision
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if singing_eer >= 0.45:
        print("\nSEVERE DOMAIN SHIFT")
        print("STRONG RECOMMENDATION: Implement DANN")
        print(f"   Expected approximately 20 point improvement")
    elif singing_eer >= 0.35:
        print("\nMODERATE DOMAIN SHIFT")
        print("RECOMMENDATION: Implement DANN")
        print(f"   Expected approximately 12-15 point improvement")
    elif singing_eer >= 0.25:
        print("\nMINOR DOMAIN SHIFT")
        print("SUGGESTION: Consider DANN")
        print(f"   Expected approximately 8-12 point improvement")
    else:
        print("\nEXCELLENT TRANSFER")
        print("DANN is optional")
        print(f"   Focus on analyzing why your model works well")
    
    # Literature comparison
    print("\n" + "="*80)
    print("COMPARISON WITH LITERATURE")
    print("="*80)
    print("\nSingFake Paper (Speech to Singing):")
    print("  AASIST:              58.12%")
    print("  Spectrogram+ResNet:  51.87%")
    print("  LFCC+ResNet:         45.12%")
    print(f"\nYour 3-branch:         {singing_eer*100:.2f}%")
    
    if singing_eer < 0.45:
        print("\nYour model beats the LFCC baseline")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        'singing_eer': singing_eer,
        'speech_eer': speech_eer,
        'dann_estimates': dann,
        'recommendation': gap_analysis['recommendation'] if speech_eer else 'Test baseline first'
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {output_path}\n")


if __name__ == "__main__":
    main()