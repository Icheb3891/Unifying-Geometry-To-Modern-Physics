"""
Uzumaki Spiral Prime Number Mapper
Generates spiral visualization with primes anchored to north axis.
Gap-determined angular spacing reveals fractal symmetric structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def get_primes(n):
    """Sieve of Eratosthenes to get all primes up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]

def calculate_gap_angles(primes, max_number):
    """
    Calculate angles for each number based on prime gap distribution.
    
    Primes are anchored at 0° (north).
    Gap sizes determine angular spacing:
    - Gap of 2: 180°
    - Gap of 4: 90°
    - Gap of 6: 60°
    - Gap of 7+: 30° (predicted)
    """
    angles = {}
    prime_set = set(primes)
    
    for num in range(1, max_number + 1):
        if num in prime_set:
            angles[num] = 0  # All primes at north
        else:
            # Find surrounding primes
            lower_prime = max([p for p in primes if p < num], default=1)
            upper_prime = min([p for p in primes if p > num], default=max_number)
            
            gap = upper_prime - lower_prime
            position_in_gap = num - lower_prime
            
            # Map gap size to base angle
            if gap == 2:
                base_angle = 180
            elif gap == 4:
                base_angle = 90
            elif gap == 6:
                base_angle = 60
            elif gap >= 7:
                base_angle = 30
            else:
                base_angle = 0
            
            # Distribute positions evenly within the gap
            angle_step = base_angle / gap
            angles[num] = position_in_gap * angle_step
    
    return angles

def spiral_coordinates(angle_deg, radius_multiplier, spiral_tightness=0.1):
    """
    Convert angle and radius multiplier to cartesian coordinates on uzumaki spiral.
    
    angle_deg: angle in degrees (0 = north)
    radius_multiplier: multiplier for spiral radius growth
    spiral_tightness: controls how tight the spiral is
    """
    angle_rad = np.radians(angle_deg)
    radius = radius_multiplier * (1 + spiral_tightness * radius_multiplier)
    
    # Convert from compass bearing (0° = north) to standard math (90° = east)
    math_angle = np.pi/2 - angle_rad
    
    x = radius * np.cos(math_angle)
    y = radius * np.sin(math_angle)
    
    return x, y

def generate_visualization(max_number=118, output_file="uzumaki_spiral.png"):
    """Generate and save the uzumaki spiral visualization"""
    
    primes = get_primes(max_number)
    angles = calculate_gap_angles(primes, max_number)
    prime_set = set(primes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16), dpi=100)
    
    # Coordinates for all numbers
    x_coords = []
    y_coords = []
    labels = []
    colors = []
    sizes = []
    
    for num in range(1, max_number + 1):
        x, y = spiral_coordinates(angles[num], num)
        x_coords.append(x)
        y_coords.append(y)
        labels.append(str(num))
        
        # Color primes differently
        if num in prime_set:
            colors.append('red')
            sizes.append(150)
        else:
            colors.append('blue')
            sizes.append(80)
    
    # Plot all points
    scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Add labels for all numbers
    for i, (x, y, label) in enumerate(zip(x_coords, y_coords, labels)):
        ax.annotate(label, (x, y), fontsize=8, ha='center', va='center', weight='bold')
    
    # Draw the spiral path
    spiral_x = []
    spiral_y = []
    for num in range(1, max_number + 1):
        x, y = spiral_coordinates(angles[num], num)
        spiral_x.append(x)
        spiral_y.append(y)
    
    ax.plot(spiral_x, spiral_y, 'gray', alpha=0.3, linewidth=1, zorder=0)
    
    # Draw reference axes at key angles (0°, 60°, 90°, 120°, 180°, etc.)
    axis_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    max_radius = max_number * 1.2
    
    for angle in axis_angles:
        x1, y1 = spiral_coordinates(angle, 0)
        x2, y2 = spiral_coordinates(angle, max_radius)
        ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.2, linewidth=0.5, linestyle='--')
        
        # Highlight 60° axes (the symmetric ones)
        if angle % 60 == 0:
            x1, y1 = spiral_coordinates(angle, 0)
            x2, y2 = spiral_coordinates(angle, max_radius)
            ax.plot([x1, x2], [y1, y2], 'green', alpha=0.5, linewidth=2, linestyle='-')
    
    # Legend
    prime_patch = mpatches.Patch(color='red', label='Primes (anchored at 0°/North)')
    composite_patch = mpatches.Patch(color='blue', label='Composites (gap-determined angles)')
    ax.legend(handles=[prime_patch, composite_patch], loc='upper right', fontsize=12)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.1)
    ax.set_title(f'Uzumaki Spiral: Prime-Gap Geometry (1-{max_number})\n' + 
                 'Primes anchored North | Gap-size determines angular spacing | 60° axes highlighted',
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()
    
    return angles, primes

def analyze_symmetric_axes(max_number=118):
    """Analyze and report numbers on symmetric 60° axes"""
    primes = get_primes(max_number)
    angles = calculate_gap_angles(primes, max_number)
    
    # Group numbers by their angle
    angle_groups = {}
    for num, angle in angles.items():
        if angle not in angle_groups:
            angle_groups[angle] = []
        angle_groups[angle].append(num)
    
    print("\n=== SYMMETRIC AXIS ANALYSIS ===\n")
    print(f"Analysis through prime: {primes[-1]} (Element: {max_number})\n")
    
    # Show 60° axis alignments
    print("Numbers aligned on 60° axes (symmetric candidates):")
    for axis_angle in [0, 60, 120, 180, 240, 300]:
        numbers_on_axis = [num for num, angle in angles.items() if angle == axis_angle]
        if numbers_on_axis:
            print(f"\n  Angle {axis_angle}°: {sorted(numbers_on_axis)}")
            
            # Show differences
            sorted_nums = sorted(numbers_on_axis)
            if len(sorted_nums) > 1:
                diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
                print(f"    Differences: {diffs}")
    
    print("\n=== GAP ANALYSIS ===\n")
    for i in range(len(primes)-1):
        gap = primes[i+1] - primes[i]
        print(f"  Gap {primes[i]} → {primes[i+1]}: {gap} numbers")

if __name__ == "__main__":
    # Generate visualization through Oganesson (118)
    print("Generating Uzumaki Spiral through Periodic Table (1-118)...\n")
    angles, primes = generate_visualization(max_number=118, output_file="uzumaki_spiral_periodic_table.png")
    
    # Analyze symmetric patterns
    analyze_symmetric_axes(max_number=118)
    
    print("\n✓ Visualization complete. Check uzumaki_spiral_periodic_table.png")
