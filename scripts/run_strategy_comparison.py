"""Execute the opt-in retrospective strategy comparison; original freeze is read-only.

From the project root: python scripts/run_strategy_comparison.py
"""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.research_strategy_comparison import run_and_export

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Read-only strategy comparison; optional export outside project/')
    parser.add_argument('--output-dir',type=Path,default=None,help='Optional directory OUTSIDE project/ for CSV/figure export')
    args=parser.parse_args()
    m=run_and_export(ROOT,args.output_dir)
    print(m[['cagr','sharpe_ratio','max_drawdown','mean_gross_exposure']].round(5).to_string())
    if args.output_dir is None:
        print('Read-only replay: no packaged artifacts modified.')
    else:
        print('Research output exported to:',args.output_dir)
