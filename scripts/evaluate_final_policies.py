"""Compare final policy alternatives in development and historical evaluation.

Run from project/: python scripts/evaluate_final_policies.py
This script does not train models, rewrite frozen artifacts or use OOS
observations to calculate development scores.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.development_policy_selection import print_final_policy_report
if __name__ == '__main__':
    print_final_policy_report(ROOT)
