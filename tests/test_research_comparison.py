"""Post-selection research regression tests. Frozen files are not rewritten."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.research_strategy_comparison import equal_active_targets, build_comparison
from src.utils import ROOT_DIR


class ResearchComparisonTests(unittest.TestCase):
    def test_equal_active_one_etf_is_capped_and_cash_retained(self):
        d=pd.to_datetime(['2022-01-07','2022-01-14'])
        s={a:pd.DataFrame({'signal':[0,0],'confidence':[.9,.9]},index=d) for a in ('SPY','IEF','GLD','SHY')}
        w=equal_active_targets(s,('SPY','IEF','GLD'))
        self.assertTrue(np.allclose(w['SPY'],.5))
        self.assertTrue(np.allclose(w.sum(axis=1),.5))

    def test_equal_active_two_active_reweights_to_full_investment(self):
        d=pd.to_datetime(['2022-01-07','2022-01-14'])
        s={a:pd.DataFrame({'signal':[0,0],'confidence':[.8,.8]},index=d) for a in ('SPY','IEF','GLD','SHY')}
        s['IEF']['signal']=1
        w=equal_active_targets(s,('SPY','IEF','GLD'))
        self.assertTrue(np.allclose(w[['SPY','IEF']].to_numpy(),.5))
        self.assertTrue(np.allclose(w.sum(axis=1),1.0))

    def test_reject_bad_signal(self):
        d=pd.to_datetime(['2022-01-07'])
        s={a:pd.DataFrame({'signal':[0],'confidence':[.8]},index=d) for a in ('SPY','IEF','GLD','SHY')}
        s['IEF'].loc[d[0],'signal']=8
        with self.assertRaises(ValueError):
            equal_active_targets(s,('SPY','IEF'))

    def test_full_comparison_reconciles_official_and_sma200(self):
        metrics, nav, targets, meta=build_comparison(ROOT_DIR)
        self.assertEqual(len(nav),1193)
        self.assertEqual(meta['stress_decisions'],58)
        self.assertEqual(meta['total_decisions'],249)
        self.assertEqual(set(targets),{'Core35_Satellite65','Equal_Active_3ETF','Equal_Active_4ETF','Regime_Adaptive_SMA200','SPY_BuyHold','Sixty_Forty'})
        self.assertAlmostEqual(float(metrics.at['Regime_Adaptive_SMA200','cagr']),.09805,delta=.000025)
        self.assertAlmostEqual(float(metrics.at['Regime_Adaptive_SMA200','sharpe_ratio']),.912,delta=.002)
        self.assertAlmostEqual(float(metrics.at['Core35_Satellite65','cagr']),.05871946082654356,places=10)
        self.assertAlmostEqual(float(metrics.at['Equal_Active_3ETF','cagr']),.053909,delta=.000025)
        self.assertLess(float(metrics.at['Regime_Adaptive_SMA200','max_drawdown']),0)

if __name__=='__main__':unittest.main()
