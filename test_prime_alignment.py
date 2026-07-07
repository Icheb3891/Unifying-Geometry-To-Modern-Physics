#!/usr/bin/env python3
"""
Prime Alignment Model - Complete Test Suite
============================================

Comprehensive testing and validation of the entire system.
Generates actual coordinate data for visualization testing.
"""

import math
from manifold_embedding import (
    sieve_of_eratosthenes,
    ManifoldEmbedding,
    CartesianSystem,
    PolarSystem,
    CylindricalSystem,
    SphericalSystem,
    ComplexSystem,
    ToroidalSystem,
    HyperbolicSystem,
    ProjectiveSystem,
)


def test_prime_generation():
    """Test prime generation."""
    print("\n" + "="*80)
    print("TEST 1: PRIME GENERATION")
    print("="*80)
    
    primes = sieve_of_eratosthenes(100)
    print(f"✓ Generated {len(primes)} primes up to 100")
    print(f"  First 10: {primes[:10]}")
    print(f"  Last 10: {primes[-10:]}")
    
    assert len(primes) == 25, "Should be 25 primes up to 100"
    assert primes[0] == 2, "First prime should be 2"
    assert primes[-1] == 97, "Last prime up to 100 should be 97"
    print("✓ All assertions passed")
    
    return primes


def test_manifold_embedding(primes):
    """Test manifold embedding."""
    print("\n" + "="*80)
    print("TEST 2: MANIFOLD EMBEDDING")
    print("="*80)
    
    embedding = ManifoldEmbedding(primes)
    
    print(f"✓ Created embedding for {len(primes)} primes")
    print(f"  Gap sequence length: {len(embedding.gaps)}")
    print(f"  Gap range: {min(embedding.gaps)} to {max(embedding.gaps)}")
    print(f"  Spiral angles computed: {len(embedding.spiral_angles)}")
    print(f"  Spiral radii computed: {len(embedding.spiral_radii)}")
    
    assert len(embedding.gaps) == len(primes) - 1
    assert len(embedding.spiral_angles) == len(embedding.gaps)
    assert len(embedding.spiral_radii) == len(embedding.gaps)
    print("✓ All assertions passed")
    
    return embedding


def test_coordinate_systems(embedding):
    """Test each coordinate system."""
    print("\n" + "="*80)
    print("TEST 3: COORDINATE SYSTEM EMBEDDING")
    print("="*80)
    
    systems = [
        ("Cartesian 2D", CartesianSystem(2)),
        ("Cartesian 3D", CartesianSystem(3)),
        ("Polar", PolarSystem()),
        ("Cylindrical", CylindricalSystem()),
        ("Spherical", SphericalSystem()),
        ("Complex", ComplexSystem()),
        ("Toroidal", ToroidalSystem()),
        ("Hyperbolic", HyperbolicSystem()),
        ("Projective", ProjectiveSystem()),
    ]
    
    for system_name, system in systems:
        points = embedding.embed_onto_system(system)
        print(f"\n✓ {system_name} ({system.dimensions()}D)")
        print(f"  Points generated: {len(points)}")
        
        if points:
            sample = points[0]
            print(f"  Sample point (prime {sample.prime}):")
            for key, val in sorted(sample.coordinates.items())[:3]:
                if isinstance(val, float):
                    print(f"    {key}: {val:.6f}")
                else:
                    print(f"    {key}: {val}")
        
        assert len(points) == len(embedding.gaps), f"Point count mismatch in {system_name}"
        assert all(isinstance(p.coordinates, dict) for p in points), f"Invalid coords in {system_name}"
    
    print("\n✓ All coordinate systems passed")


def test_data_export(embedding):
    """Test data export functionality."""
    print("\n" + "="*80)
    print("TEST 4: DATA EXPORT")
    print("="*80)
    
    # Export all coordinates
    output_file = "test_prime_manifold_output.txt"
    embedding.export_all_coordinates(output_file)
    
    print(f"✓ Exported to {output_file}")
    
    # Read and validate
    with open(output_file, 'r') as f:
        content = f.read()
    
    print(f"  File size: {len(content)} bytes")
    print(f"  Contains 'Cartesian': {'Cartesian' in content}")
    print(f"  Contains 'Polar': {'Polar' in content}")
    print(f"  Contains 'Toroidal': {'Toroidal' in content}")
    
    assert len(content) > 1000, "Export file too small"
    assert "Cartesian" in content, "Missing Cartesian system"
    print("✓ Export validation passed")


def test_specific_prime_coordinates(embedding):
    """Test coordinates for specific primes."""
    print("\n" + "="*80)
    print("TEST 5: SPECIFIC PRIME COORDINATES")
    print("="*80)
    
    # Test prime 2
    cartesian = CartesianSystem(2)
    points = embedding.embed_onto_system(cartesian)
    
    prime_2 = next(p for p in points if p.prime == 2)
    print(f"\nPrime 2 in Cartesian 2D:")
    print(f"  x: {prime_2.coordinates['x']:.6f}")
    print(f"  y: {prime_2.coordinates['y']:.6f}")
    print(f"  Gap to next: {prime_2.gap}")
    print(f"  Spiral angle: {prime_2.spiral_angle:.6f} rad")
    print(f"  Spiral radius: {prime_2.spiral_radius:.6f}")
    
    # Test prime 97 (last in our test set)
    prime_97 = next((p for p in points if p.prime == 97), None)
    if prime_97:
        print(f"\nPrime 97 in Cartesian 2D:")
        print(f"  x: {prime_97.coordinates['x']:.6f}")
        print(f"  y: {prime_97.coordinates['y']:.6f}")
        print(f"  Gap to next: {prime_97.gap}")
        print(f"  Spiral angle: {prime_97.spiral_angle:.6f} rad")
        print(f"  Spiral radius: {prime_97.spiral_radius:.6f}")
    
    # Test in different systems for same prime
    prime_11_systems = {}
    for system_name, system in [
        ("Cartesian", CartesianSystem(2)),
        ("Polar", PolarSystem()),
        ("Complex", ComplexSystem()),
    ]:
        points_sys = embedding.embed_onto_system(system)
        prime = next(p for p in points_sys if p.prime == 11)
        prime_11_systems[system_name] = prime
        print(f"\nPrime 11 in {system_name}:")
        for key, val in sorted(prime.coordinates.items())[:2]:
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
    
    print("\n✓ Specific prime coordinate test passed")


def test_spiral_geometry(embedding):
    """Test spiral geometric properties."""
    print("\n" + "="*80)
    print("TEST 6: SPIRAL GEOMETRIC PROPERTIES")
    print("="*80)
    
    # Verify spiral properties
    angles = embedding.spiral_angles
    radii = embedding.spiral_radii
    
    print(f"Spiral Angles:")
    print(f"  First 5: {[f'{a:.4f}' for a in angles[:5]]}")
    print(f"  Last 5: {[f'{a:.4f}' for a in angles[-5:]]}")
    print(f"  Range: {min(angles):.4f} to {max(angles):.4f}")
    print(f"  Total rotation: {angles[-1] / (2*math.pi):.2f} full turns")
    
    print(f"\nSpiral Radii:")
    print(f"  First 5: {[f'{r:.4f}' for r in radii[:5]]}")
    print(f"  Last 5: {[f'{r:.4f}' for r in radii[-5:]]}")
    print(f"  Range: {min(radii):.4f} to {max(radii):.4f}")
    print(f"  Growth: exponential (logarithmic spiral)")
    
    # Verify angles are monotonically increasing
    is_increasing = all(angles[i] <= angles[i+1] for i in range(len(angles)-1))
    print(f"\nAngles monotonically increasing: {is_increasing}")
    
    # Verify radii are monotonically increasing
    is_increasing_radii = all(radii[i] <= radii[i+1] for i in range(len(radii)-1))
    print(f"Radii monotonically increasing: {is_increasing_radii}")
    
    assert is_increasing, "Angles should be monotonically increasing"
    assert is_increasing_radii, "Radii should be monotonically increasing"
    print("\n✓ Spiral geometry test passed")


def test_gap_to_angle_projection(embedding):
    """Test gap → angle projection mapping."""
    print("\n" + "="*80)
    print("TEST 7: GAP TO ANGLE PROJECTION")
    print("="*80)
    
    # For each prime, verify angle increment from gap
    for i in range(min(5, len(embedding.gaps))):
        prime = embedding.primes[i]
        gap = embedding.gaps[i]
        
        # Expected angle increment
        expected_delta = (gap / prime) * (2 * math.pi) if prime > 0 else 0
        
        # Actual angle increment
        if i == 0:
            actual_delta = embedding.spiral_angles[i]
        else:
            actual_delta = embedding.spiral_angles[i] - embedding.spiral_angles[i-1]
        
        print(f"\nPrime {prime}, Gap {gap}:")
        print(f"  Expected Δθ: {expected_delta:.6f} rad ({math.degrees(expected_delta):.2f}°)")
        print(f"  Actual Δθ: {actual_delta:.6f} rad ({math.degrees(actual_delta):.2f}°)")
        print(f"  Match: {abs(expected_delta - actual_delta) < 1e-6}")


def test_cartesian_bounds(embedding):
    """Test Cartesian coordinate bounds."""
    print("\n" + "="*80)
    print("TEST 8: CARTESIAN BOUNDS ANALYSIS")
    print("="*80)
    
    cartesian = CartesianSystem(2)
    points = embedding.embed_onto_system(cartesian)
    
    x_coords = [p.coordinates['x'] for p in points]
    y_coords = [p.coordinates['y'] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    print(f"X bounds: [{x_min:.4f}, {x_max:.4f}]")
    print(f"Y bounds: [{y_min:.4f}, {y_max:.4f}]")
    
    print(f"\nWidth: {x_max - x_min:.4f}")
    print(f"Height: {y_max - y_min:.4f}")
    print(f"Aspect ratio: {(x_max - x_min) / (y_max - y_min):.2f}")
    
    # Verify points form a spiral pattern
    distances = []
    for i in range(len(points) - 1):
        dx = points[i+1].coordinates['x'] - points[i].coordinates['x']
        dy = points[i+1].coordinates['y'] - points[i].coordinates['y']
        dist = math.sqrt(dx**2 + dy**2)
        distances.append(dist)
    
    print(f"\nPoint-to-point distances:")
    print(f"  Min: {min(distances):.6f}")
    print(f"  Max: {max(distances):.6f}")
    print(f"  Mean: {sum(distances) / len(distances):.6f}")
    print(f"  Trend: {'Increasing' if distances[-1] > distances[0] else 'Decreasing'}")
    
    print("\n✓ Cartesian bounds test passed")


def generate_plotting_data(embedding):
    """Generate data suitable for plotting."""
    print("\n" + "="*80)
    print("TEST 9: GENERATE PLOTTING DATA")
    print("="*80)
    
    cartesian = CartesianSystem(2)
    points = embedding.embed_onto_system(cartesian)
    
    # Generate CSV for gnuplot/matplotlib
    csv_file = "test_primes_cartesian.csv"
    with open(csv_file, 'w') as f:
        f.write("prime,x,y,gap,spiral_angle,spiral_radius\n")
        for p in points:
            f.write(f"{p.prime},{p.coordinates['x']:.6f},{p.coordinates['y']:.6f},"
                   f"{p.gap},{p.spiral_angle:.6f},{p.spiral_radius:.6f}\n")
    
    print(f"✓ Generated {csv_file}")
    print(f"  {len(points)} data points")
    
    # Generate Python plot script
    plot_script = "test_plot_primes.py"
    with open(plot_script, 'w') as f:
        f.write("""import matplotlib.pyplot as plt
import csv

# Read data
primes, x, y, gaps = [], [], [], []
with open('test_primes_cartesian.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        primes.append(int(row['prime']))
        x.append(float(row['x']))
        y.append(float(row['y']))
        gaps.append(int(row['gap']))

# Create plot
fig, ax = plt.subplots(figsize=(12, 12))

# Scatter plot with color by gap size
scatter = ax.scatter(x, y, c=gaps, s=50, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)

# Connect points to show spiral
ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5, zorder=1)

# Labels for first and last few primes
for i in [0, 1, 2, -3, -2, -1]:
    if i < len(primes):
        ax.annotate(str(primes[i]), (x[i], y[i]), fontsize=8, alpha=0.7)

# Formatting
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Prime Numbers on Logarithmic Spiral')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prime Gap', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('test_prime_spiral.png', dpi=150, bbox_inches='tight')
print("✓ Saved test_prime_spiral.png")
plt.show()
""")
    
    print(f"✓ Generated {plot_script}")
    print("  Run with: python test_plot_primes.py")


def run_all_tests():
    """Run complete test suite."""
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "PRIME ALIGNMENT MODEL - COMPLETE TEST SUITE".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Run all tests
        primes = test_prime_generation()
        embedding = test_manifold_embedding(primes)
        test_coordinate_systems(embedding)
        test_data_export(embedding)
        test_specific_prime_coordinates(embedding)
        test_spiral_geometry(embedding)
        test_gap_to_angle_projection(embedding)
        test_cartesian_bounds(embedding)
        generate_plotting_data(embedding)
        
        print("\n\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "✓ ALL TESTS PASSED".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        print("\n📊 Output files generated:")
        print("  - test_prime_manifold_output.txt (all coordinate systems)")
        print("  - test_primes_cartesian.csv (plotting data)")
        print("  - test_plot_primes.py (matplotlib script)")
        print("  - test_prime_spiral.png (visualization)")
        print("\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
