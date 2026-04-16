"""
Author: Krushna Sanjay Sharma
Description: Master pipeline entry point for Pixels to Prognosis v2.
This script dispatches to the distinct tasks (1, 2, 3, or all).
"""

import argparse
import sys


def parse_args():
    """Parses command-line arguments for task dispatching."""
    parser = argparse.ArgumentParser(
        description="Pixels to Prognosis v2 - Pipeline Run",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Which task to run: 1 (Transfer ViT), 2 (Grad-CAM), 3 (Gabor ResNet), or all."
    )
    
    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args, _ = parser.parse_known_args()
    return args, parser


def run_task1():
    """Delegates to Task 1 orchestrator."""
    print("--- Running Task 1: Transfer ViT ---")
    import tasks.task1_transfer_vit as t1
    t1.main()

def run_task2():
    """Delegates to Task 2 orchestrator."""
    print("--- Running Task 2: Grad-CAM Analysis ---")
    import tasks.task2_gradcam as t2
    t2.main()

def run_task3():
    """Delegates to Task 3 orchestrator."""
    print("--- Running Task 3: Gabor ResNet ---")
    import tasks.task3_gabor_resnet as t3
    t3.main()


def main():
    """Main execution point."""
    args, parser = parse_args()
    
    if args.task == "1":
        run_task1()
    elif args.task == "2":
        run_task2()
    elif args.task == "3":
        run_task3()
    elif args.task == "all":
        run_task1()
        run_task2()
        run_task3()
    else:
        print(f"Error: Invalid task '{args.task}' specified.\n")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
