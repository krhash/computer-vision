# main.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Master pipeline that runs all five project tasks in order.
#              Each task can also be run independently via its own file in
#              the tasks/ directory. This file exists for full end-to-end
#              reproduction (e.g. grading).
#
# Usage:
#   python main.py               # runs all tasks
#   python main.py --task 1      # runs only Task 1
#   python main.py --task 1 2    # runs Tasks 1 and 2

import sys
import argparse


def parse_args(argv: list) -> argparse.Namespace:
    """
    Parses command-line arguments for the master pipeline.

    Args:
        argv (list): sys.argv list.

    Returns:
        argparse.Namespace with a 'tasks' attribute (list of ints).
    """
    parser = argparse.ArgumentParser(
        description="Project 5 — Recognition using Deep Networks"
    )
    parser.add_argument(
        "--task",
        dest    = "tasks",
        nargs   = "*",
        type    = int,
        default = [1, 2, 3, 4, 5],
        help    = "Task number(s) to run. Default: all tasks (1–5).",
    )
    return parser.parse_args(argv[1:])


def run_task(task_num: int) -> None:
    """
    Dispatches execution to the appropriate task module.

    Each task module exposes a main(argv) function consistent with the
    project structure convention.

    Args:
        task_num (int): Task number to execute (1–5).
    """
    print(f"\n{'=' * 60}")
    print(f"  RUNNING TASK {task_num}")
    print(f"{'=' * 60}\n")

    if task_num == 1:
        from tasks.task1_build_train import main as task1_main
        task1_main([])

    elif task_num == 2:
        # Task 2 will be implemented in a later step
        from tasks.task2_examine import main as task2_main
        task2_main([])

    elif task_num == 3:
        # Task 3 will be implemented in a later step
        from tasks.task3_greek import main as task3_main
        task3_main([])

    elif task_num == 4:
        # Task 4 will be implemented in a later step
        from tasks.task4_transformer import main as task4_main
        task4_main([])

    elif task_num == 5:
        # Task 5 will be implemented in a later step
        from tasks.task5_experiment import main as task5_main
        task5_main([])

    else:
        print(f"  [WARNING] Unknown task number: {task_num}. Skipping.")


def main(argv: list) -> None:
    """
    Master entry point. Parses which tasks to run and dispatches them.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Project 5: Recognition using Deep Networks")
    print(f"  Tasks to run: {args.tasks}")
    print("=" * 60)

    for task_num in sorted(args.tasks):
        run_task(task_num)

    print("\n  All requested tasks complete.")


if __name__ == "__main__":
    main(sys.argv)