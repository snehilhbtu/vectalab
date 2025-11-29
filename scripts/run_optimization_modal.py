import sys
import os
import modal
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectalab.core import Vectalab

def run_optimization():
    # Enable Modal logs
    modal.enable_output()
    
    input_path = "test_data/png_multi/google.png"
    output_path = "test_data/vectalab_multi/google_modal_bayesian.svg"
    
    print(f"Starting Bayesian optimization with Modal SAM on {input_path}...")
    
    try:
        # Initialize Vectalab with Bayesian method and Modal backend
        vl = Vectalab(
            method="bayesian",
            model_type="vit_h",  # Use the huge model on cloud!
            use_modal=True,
            device="cpu" # Local device for Bayesian part (or cuda if available locally)
        )
        
        # Run vectorization
        vl.vectorize(input_path, output_path)
        
        print(f"Successfully created {output_path}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimization()
