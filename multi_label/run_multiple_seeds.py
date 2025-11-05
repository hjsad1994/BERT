"""
Run Multi-Label Training with Multiple Seeds
Executes training multiple times with different seeds and aggregates metrics
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
from typing import List, Dict, Any
import logging

# Seeds to run
SEEDS = [100, 101, 102, 103, 104]

# Output directory for aggregated results
AGGREGATED_OUTPUT_DIR = "multi_label/results/multi_seed_experiments"

def setup_logging(output_dir: str) -> None:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'multi_seed_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Multi-seed experiment log: {log_file}")

def update_config_seed(config_path: str, seed: int) -> None:
    """Update training seed in config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['reproducibility']['training_seed'] = seed
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logging.info(f"Updated config with seed: {seed}")

def run_training(seed: int, config_path: str, run_output_dir: str) -> bool:
    """Run training for a single seed"""
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting training with seed: {seed}")
    logging.info(f"Output directory: {run_output_dir}")
    logging.info(f"{'='*80}\n")
    
    # Update config with new seed
    update_config_seed(config_path, seed)
    
    # Run training script
    cmd = [
        sys.executable,  # Current Python interpreter
        "multi_label/train_multilabel.py",
        "--config", config_path,
        "--output-dir", run_output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        logging.info(f"Training completed successfully for seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed for seed {seed}: {e}")
        return False

def collect_results(seed: int, run_output_dir: str) -> Dict[str, Any]:
    """Collect results from a training run"""
    results = {
        'seed': seed,
        'run_dir': run_output_dir
    }
    
    # Load test results JSON
    test_results_file = os.path.join(run_output_dir, 'test_results.json')
    if os.path.exists(test_results_file):
        with open(test_results_file, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
        
        # Extract overall metrics
        results['test_accuracy'] = test_results['test_accuracy']
        results['test_f1'] = test_results['test_f1']
        results['test_precision'] = test_results['test_precision']
        results['test_recall'] = test_results['test_recall']
        
        # Extract per-aspect metrics
        for aspect, metrics in test_results['per_aspect'].items():
            results[f'{aspect}_accuracy'] = metrics['accuracy']
            results[f'{aspect}_f1'] = metrics['f1']
            results[f'{aspect}_precision'] = metrics['precision']
            results[f'{aspect}_recall'] = metrics['recall']
        
        logging.info(f"Collected results for seed {seed}:")
        logging.info(f"  Accuracy: {results['test_accuracy']*100:.2f}%")
        logging.info(f"  F1 Score: {results['test_f1']*100:.2f}%")
    else:
        logging.error(f"Test results not found for seed {seed}: {test_results_file}")
    
    # Load training history
    history_file = os.path.join(run_output_dir, 'training_history.csv')
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file, encoding='utf-8-sig')
        results['training_history'] = history_df.to_dict('records')
        results['best_val_f1'] = history_df['val_f1'].max()
        results['best_val_accuracy'] = history_df['val_accuracy'].max()
    
    return results

def aggregate_metrics(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Aggregate metrics across all runs"""
    logging.info(f"\n{'='*80}")
    logging.info("Aggregating Metrics Across All Seeds")
    logging.info(f"{'='*80}\n")
    
    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(all_results)
    
    # Columns to aggregate (all numeric columns except seed)
    metric_columns = [col for col in df_results.columns 
                     if col not in ['seed', 'run_dir', 'training_history']]
    
    # Calculate mean and std for each metric
    aggregated = {}
    for col in metric_columns:
        if col in df_results.columns:
            values = df_results[col].dropna()
            if len(values) > 0:
                aggregated[f'{col}_mean'] = values.mean()
                aggregated[f'{col}_std'] = values.std()
    
    # Save individual run results
    individual_results_file = os.path.join(output_dir, 'individual_runs_results.csv')
    df_results.drop(columns=['training_history'], errors='ignore').to_csv(
        individual_results_file, index=False, encoding='utf-8-sig'
    )
    logging.info(f"Individual run results saved to: {individual_results_file}")
    
    # Save aggregated statistics
    aggregated_df = pd.DataFrame([aggregated])
    aggregated_file = os.path.join(output_dir, 'aggregated_statistics.csv')
    aggregated_df.to_csv(aggregated_file, index=False, encoding='utf-8-sig')
    logging.info(f"Aggregated statistics saved to: {aggregated_file}")
    
    # Create detailed report
    report_file = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-Seed Training Experiment Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seeds Used: {SEEDS}\n")
        f.write(f"Number of Runs: {len(all_results)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL METRICS (Mean ± Std)\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        overall_metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
        for metric in overall_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in aggregated and std_key in aggregated:
                mean_val = aggregated[mean_key] * 100
                std_val = aggregated[std_key] * 100
                f.write(f"{metric.replace('test_', '').upper():<15}: {mean_val:6.2f}% ± {std_val:5.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-ASPECT METRICS (Mean ± Std)\n")
        f.write("=" * 80 + "\n\n")
        
        # Find all aspects
        aspects = set()
        for col in metric_columns:
            if '_accuracy' in col or '_f1' in col:
                aspect = col.replace('_accuracy', '').replace('_f1', '').replace('_precision', '').replace('_recall', '')
                if aspect not in ['test', 'best_val']:
                    aspects.add(aspect)
        
        aspects = sorted(aspects)
        
        # Per-aspect metrics
        f.write(f"{'Aspect':<15} {'Accuracy':<20} {'F1 Score':<20} {'Precision':<20} {'Recall':<20}\n")
        f.write("-" * 95 + "\n")
        
        for aspect in aspects:
            acc_mean = aggregated.get(f'{aspect}_accuracy_mean', 0) * 100
            acc_std = aggregated.get(f'{aspect}_accuracy_std', 0) * 100
            f1_mean = aggregated.get(f'{aspect}_f1_mean', 0) * 100
            f1_std = aggregated.get(f'{aspect}_f1_std', 0) * 100
            prec_mean = aggregated.get(f'{aspect}_precision_mean', 0) * 100
            prec_std = aggregated.get(f'{aspect}_precision_std', 0) * 100
            rec_mean = aggregated.get(f'{aspect}_recall_mean', 0) * 100
            rec_std = aggregated.get(f'{aspect}_recall_std', 0) * 100
            
            f.write(f"{aspect:<15} {acc_mean:5.2f}±{acc_std:4.2f}%  {f1_mean:5.2f}±{f1_std:4.2f}%  ")
            f.write(f"{prec_mean:5.2f}±{prec_std:4.2f}%  {rec_mean:5.2f}±{rec_std:4.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INDIVIDUAL RUN RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"Seed {result['seed']}:\n")
            f.write(f"  Accuracy:  {result.get('test_accuracy', 0)*100:.2f}%\n")
            f.write(f"  F1 Score:  {result.get('test_f1', 0)*100:.2f}%\n")
            f.write(f"  Precision: {result.get('test_precision', 0)*100:.2f}%\n")
            f.write(f"  Recall:    {result.get('test_recall', 0)*100:.2f}%\n")
            f.write(f"  Output Dir: {result['run_dir']}\n\n")
    
    logging.info(f"Detailed report saved to: {report_file}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80 + "\n")
    print(f"Seeds: {SEEDS}")
    print(f"Runs: {len(all_results)}\n")
    print("Overall Metrics (Mean ± Std):")
    for metric in overall_metrics:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        if mean_key in aggregated and std_key in aggregated:
            mean_val = aggregated[mean_key] * 100
            std_val = aggregated[std_key] * 100
            print(f"  {metric.replace('test_', '').upper():<15}: {mean_val:6.2f}% ± {std_val:5.2f}%")
    
    print(f"\nDetailed results saved to: {output_dir}")

def main():
    """Main function to run multi-seed experiments"""
    print("=" * 80)
    print("Multi-Seed Training Experiment")
    print("=" * 80)
    print(f"\nSeeds to run: {SEEDS}")
    print(f"Output directory: {AGGREGATED_OUTPUT_DIR}\n")
    
    # Setup logging
    os.makedirs(AGGREGATED_OUTPUT_DIR, exist_ok=True)
    setup_logging(AGGREGATED_OUTPUT_DIR)
    
    config_path = "multi_label/config_multi.yaml"
    
    # Backup original config
    backup_config = config_path + ".backup"
    shutil.copy2(config_path, backup_config)
    logging.info(f"Backed up config to: {backup_config}")
    
    all_results = []
    
    try:
        for i, seed in enumerate(SEEDS, 1):
            print(f"\n{'='*80}")
            print(f"Run {i}/{len(SEEDS)} - Seed: {seed}")
            print(f"{'='*80}\n")
            
            # Create output directory for this run
            run_output_dir = os.path.join(AGGREGATED_OUTPUT_DIR, f"seed_{seed}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            # Run training
            success = run_training(seed, config_path, run_output_dir)
            
            if success:
                # Collect results
                results = collect_results(seed, run_output_dir)
                all_results.append(results)
                
                # Save intermediate results
                intermediate_file = os.path.join(AGGREGATED_OUTPUT_DIR, 'intermediate_results.json')
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            else:
                logging.error(f"Skipping seed {seed} due to training failure")
        
        # Aggregate all results
        if all_results:
            aggregate_metrics(all_results, AGGREGATED_OUTPUT_DIR)
            
            print("\n" + "=" * 80)
            print("ALL RUNS COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"\nResults directory: {AGGREGATED_OUTPUT_DIR}")
            print("\nGenerated files:")
            print(f"  - individual_runs_results.csv: Results for each seed")
            print(f"  - aggregated_statistics.csv: Mean and std for all metrics")
            print(f"  - experiment_report.txt: Detailed report for paper")
        else:
            logging.error("No successful runs to aggregate")
    
    finally:
        # Restore original config
        shutil.copy2(backup_config, config_path)
        logging.info(f"Restored original config from: {backup_config}")
        print(f"\nOriginal config restored")

if __name__ == '__main__':
    main()
