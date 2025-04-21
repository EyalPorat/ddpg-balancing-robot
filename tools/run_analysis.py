"""
Script to run model analysis on a trained balancing robot model.
"""

import os
import sys
import argparse
from pathlib import Path

# Import the ModelAnalyzer class
from model_analyzer import ModelAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Run analysis on trained balancing robot model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env-config", type=str, required=True, help="Path to environment config file")
    parser.add_argument("--ddpg-config", type=str, help="Path to DDPG config file")
    parser.add_argument("--output", type=str, default="analysis_results", help="Output directory")
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "basic", "performance", "comparison", "trajectory"],
        help="Type of analysis to run",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running analysis on model: {args.model}")
    print(f"Using environment config: {args.env_config}")
    if args.ddpg_config:
        print(f"Using DDPG config: {args.ddpg_config}")
    print(f"Saving results to: {args.output}")

    # Create model analyzer
    analyzer = ModelAnalyzer(
        model_path=args.model,
        env_config_path=args.env_config,
        ddpg_config_path=args.ddpg_config,
        output_dir=args.output,
        device=args.device,
    )

    # Run appropriate analysis
    if args.type == "all":
        analyzer.run_complete_analysis()
    elif args.type == "basic":
        analyzer.analyze_response_curves()
        analyzer.create_action_heatmap()
        analyzer.create_phase_space_plot()
    elif args.type == "performance":
        analyzer.analyze_simulated_trajectories()
        analyzer.analyze_stability_regions()
    elif args.type == "comparison":
        analyzer.generate_comparative_pd_controller()
        analyzer.analyze_controller_nonlinearity()
    elif args.type == "trajectory":
        analyzer.analyze_simulated_trajectories()


    print(f"Analysis complete. Results saved to {args.output}")
    print(f"Open {output_dir / 'analysis_summary.html'} in a browser to view results")


if __name__ == "__main__":
    main()
