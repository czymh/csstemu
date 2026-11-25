"""Compare direct xi_hh after changing cosmology with fresh instances."""
import gc
import numpy as np
from test_regression_snapshot import COSMOS, Z_LOW, R_GRID
from CEmulator.Emulator import Xihh_CEmulator


def predict(emu):
    return emu.get_xihh_lgnbar_threshold(
        z=Z_LOW[:2], r=R_GRID[:8], lgden1=-3.5, lgden2=-3.0,
        massdef='RockstarM200m')


def main():
    seq = Xihh_CEmulator(verbose=False)
    for name in ('c0', 'c2', 'c1', 'c0'):
        seq.set_cosmos(**COSMOS[name])
        actual = predict(seq)
        fresh = Xihh_CEmulator(verbose=False)
        fresh.set_cosmos(**COSMOS[name])
        expected = predict(fresh)
        assert np.all(np.isfinite(actual)), name
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print('%s: direct xi_hh sequential/fresh PASS' % name, flush=True)
        del fresh
        gc.collect()


if __name__ == '__main__':
    main()
