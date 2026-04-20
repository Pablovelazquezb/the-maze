"""
Main script to run entire second check-in
Run this to train, evaluate, and test everything
COSC 4368 - Spring 2026
"""

import os
import sys
import subprocess
import json
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout[-500:])  # Last 500 chars of output
    else:
        print(f"❌ {description} failed with error:")
        print(result.stderr)
        return False
    return True


def check_files_exist(files):
    """Check if required files exist"""
    missing = []
    for f in files:
        if not os.path.exists(f):
            missing.append(f)
    return missing


def main():
    print("=" * 70)
    print("SILENT CARTOGRAPHER - SECOND CHECK-IN")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check for required files
    required_files = ['environment.py', 'ql_agent.py', 'metrics.py']
    missing = check_files_exist(required_files)
    
    if missing:
        print("\n❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease create these files before running.")
        return
    
    # Check for maze-beta
    if not os.path.exists('maze_beta.txt'):
        print("\n⚠️  WARNING: maze_beta.txt not found!")
        print("   You need to get maze-beta from professor/TA.")
        print("   Continuing with training on maze-alpha only...")
    
    print("\n" + "=" * 70)
    print("STEP 1: Training on Maze-Alpha")
    print("=" * 70)
    
    # Run training
    success = run_command(
        "python train_maze_alpha.py",
        "Training Q-Learning agent on maze-alpha"
    )
    
    if not success:
        print("\n❌ Training failed. Check errors above.")
        return
    
    print("\n" + "=" * 70)
    print("STEP 2: Testing on Maze-Beta (NO TRAINING)")
    print("=" * 70)
    
    # Run maze-beta test if file exists
    if os.path.exists('maze_beta.txt'):
        success = run_command(
            "python test_maze_beta.py",
            "Testing trained agent on maze-beta"
        )
    else:
        print("\n⚠️  Skipping maze-beta test - file not found")
        print("   Please add maze_beta.txt and run test_maze_beta.py manually")
    
    print("\n" + "=" * 70)
    print("STEP 3: Generating Metrics Report")
    print("=" * 70)
    
    # Generate metrics report
    try:
        # Load training metrics
        if os.path.exists('training_metrics.json'):
            with open('training_metrics.json', 'r') as f:
                training_metrics = json.load(f)
            
            print("\n✅ Training metrics loaded")
            print(f"   Final success rate: {training_metrics['success_rate'][-1]*100:.1f}%")
            print(f"   Final average turns: {training_metrics['avg_turns'][-1]:.1f}")
        
        # Load beta results
        if os.path.exists('maze_beta_results.json'):
            with open('maze_beta_results.json', 'r') as f:
                beta_results = json.load(f)
            
            print("\n✅ Maze-beta results loaded")
            print(f"   Success rate: {beta_results['success_rate']*100:.1f}%")
            print(f"   Average turns: {beta_results['avg_turns']:.1f}")
    except Exception as e:
        print(f"⚠️  Could not load metrics: {e}")
    
    print("\n" + "=" * 70)
    print("SECOND CHECK-IN COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    
    generated_files = [
        'q_table_final.pkl',      # Trained Q-table
        'training_metrics.json',   # Training metrics
        'maze_beta_results.json',  # Test results (if maze-beta exists)
        'solution_path.png'        # Visualization (if generated)
    ]
    
    for f in generated_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✅ {f} ({size:,} bytes)")
    
    print("\nNext steps:")
    print("  1. Update your report with metrics")
    print("  2. Prepare presentation slides")
    print("  3. Practice demo for April 20/22")
    print("  4. Submit all files (ZIP or GitHub)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()