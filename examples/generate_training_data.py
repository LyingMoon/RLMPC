"""Example script for generating MPC training data using Python instead of MATLAB."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_generation import DataGenerationConfig, generate_mpc_training_data
from src.mpc.neural_mpc import train_neural_mpc
from src.mpc.python_mpc import QubeSystemModel, CustomSystemModel
import argparse
import time


def main():
    """Generate training data and optionally train neural MPC."""
    parser = argparse.ArgumentParser(description="Generate MPC training data")
    parser.add_argument("--samples", type=int, default=10000, 
                       help="Number of training samples (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="data/python_generated",
                       help="Output directory for generated data")
    parser.add_argument("--train", action="store_true",
                       help="Also train neural MPC after data generation")
    parser.add_argument("--prediction-time", type=float, default=0.5,
                       help="MPC prediction horizon time (seconds)")
    parser.add_argument("--control-dt", type=float, default=0.1,
                       help="Control sample time (seconds)")
    parser.add_argument("--system-dt", type=float, default=0.01,
                       help="System sample time (seconds)")
    parser.add_argument("--system-model", type=str, default="qube",
                       choices=["qube", "custom"],
                       help="System model to use (default: qube)")

    args = parser.parse_args()

    # Map system model choice to class
    system_models = {
        "qube": QubeSystemModel,
        "custom": CustomSystemModel
    }
    
    # Configure data generation
    config = DataGenerationConfig(
        n_samples=args.samples,
        prediction_time=args.prediction_time,
        control_sample_time=args.control_dt,
        system_sample_time=args.system_dt,
        system_model_class=system_models[args.system_model],
        output_dir=args.output_dir
    )
    
    print(f"Generating {args.samples} training samples...")
    print(f"System model: {args.system_model}")
    print(f"Prediction time: {args.prediction_time}s")
    print(f"Control sample time: {args.control_dt}s")
    print(f"System sample time: {args.system_dt}s")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Generate data
    start_time = time.time()
    try:
        saved_files = generate_mpc_training_data(config)
        generation_time = time.time() - start_time
        
        print(f"Data generation completed in {generation_time:.1f} seconds")
        print("Generated files:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
        print()
        
        # Train neural MPC if requested
        if args.train:
            print("Training Neural MPC...")
            if 'input' in saved_files and 'output' in saved_files:
                results = train_neural_mpc(
                    input_path=str(saved_files['input']),
                    output_path=str(saved_files['output']),
                    save_path=f"{args.output_dir}/neural_mpc_model.pth"
                )
                print(f"Training completed. Best loss: {results['best_loss']:.6f}")
                print(f"Final train loss: {results['train_losses'][-1]:.6f}")
                print(f"Epochs: {results['epochs']}")
            else:
                print("Generated files not available for training.")
    
    except Exception as e:
        print(f"Error during data generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())