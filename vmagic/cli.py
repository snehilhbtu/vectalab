import argparse
import sys
from vmagic.core import VMagic

def main():
    parser = argparse.ArgumentParser(description="vmagic: SOTA Image Vectorizer")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output SVG")
    parser.add_argument("--model", default="vit_b", help="SAM model type (vit_h, vit_l, vit_b)")
    parser.add_argument("--device", default="cpu", help="Device to run SAM on (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    try:
        vm = VMagic(model_type=args.model, device=args.device)
        vm.vectorize(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
