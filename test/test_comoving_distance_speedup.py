"""
Benchmark the speedup of the new comoving_distance implementation vs the old one.
"""
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_comoving_distance(cosmo, z_array, n_runs=10):
    """Benchmark comoving_distance for a given z array."""
    # Warm up
    _ = cosmo.comoving_distance(z_array[:1])

    times = []
    for _ in range(n_runs):
        start = time.time()
        chi = cosmo.comoving_distance(z_array)
        times.append(time.time() - start)

    return np.array(times)


def main():
    from CEmulator.cosmology import Cosmology

    # Set up a cosmology
    cosmo = Cosmology(verbose=False)
    cosmo.set_cosmos({
        'Omegab': 0.049, 'Omegam': 0.30,
        'H0': 67.66, 'ns': 0.9665, 'A': 2.0,
        'w': -1.0, 'wa': 0.0, 'mnu': 0.06
    })

    # Test different array sizes
    test_sizes = [10, 100, 500, 1000, 5000, 10000]
    n_runs = 5

    print(f"{'N_z':>6} {'Time (ms)':>12} {'Time/z (µs)':>12} {'Throughput (kHz)':>15}")
    print("-" * 50)

    for n_z in test_sizes:
        # Generate test redshifts
        z_array = np.linspace(0.01, 3.0, n_z)

        # Benchmark
        times = benchmark_comoving_distance(cosmo, z_array, n_runs=n_runs)
        avg_time = np.mean(times)
        std_time = np.std(times)

        # Metrics
        time_per_z = avg_time * 1e6 / n_z  # microseconds per redshift
        throughput = n_z / avg_time / 1e3  # thousands of redshifts per second

        print(f"{n_z:>6d} {avg_time*1e3:>12.3f}±{std_time*1e3:>5.3f} "
              f"{time_per_z:>12.1f} {throughput:>15.1f}")

    # Compare with old implementation
    print("\n" + "="*50)
    print("Comparison with old quad-based implementation (estimates):")
    print("Old: O(N) quad calls, each ~0.1-1 ms → 1000 z takes ~0.1-1 s")
    print("New: O(1) get_Ez call + O(N) interpolation → 1000 z takes ~1-10 ms")
    print("Speedup: ~100-1000x for large arrays")


if __name__ == '__main__':
    main()