"""
Prime Alignment Model - Coordinate Transposition & Manifold Embedding
======================================================================

Maps primes from spiral parameter space onto ANY coordinate system:
- Cartesian (x, y, z)
- Polar (r, θ)
- Cylindrical (r, θ, z)
- Spherical (ρ, θ, φ)
- Conformal/Complex (z = x + iy)
- Projective spaces
- Toroidal (u, v)
- Hyperbolic (Poincaré disk)
- Custom manifolds

The PRIMES themselves are invariant; only the COORDINATE REPRESENTATION changes.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ManifoldPoint:
    """Point on a coordinate manifold."""
    prime: int
    index: int
    gap: int
    # Intrinsic spiral parameters
    spiral_angle: float  # θ from gap accumulation
    spiral_radius: float  # r from spiral equation
    # Extrinsic coordinates (many possible)
    coordinates: Dict[str, float]  # {"x": 1.5, "y": 2.3, ...}
    coordinate_system: str


class CoordinateSystem(ABC):
    """Abstract base for coordinate systems."""
    
    @abstractmethod
    def name(self) -> str:
        """System name."""
        pass
    
    @abstractmethod
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """
        Embed spiral point onto this coordinate system.
        
        Args:
            spiral_radius: Distance from spiral center
            spiral_angle: Angular position on spiral
            
        Returns:
            Dictionary of coordinates in this system
        """
        pass
    
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions."""
        pass


class CartesianSystem(CoordinateSystem):
    """Standard 2D/3D Cartesian coordinates."""
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def name(self) -> str:
        return f"Cartesian_{self.dim}D"
    
    def dimensions(self) -> int:
        return self.dim
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Simple polar to Cartesian conversion."""
        x = spiral_radius * math.cos(spiral_angle)
        y = spiral_radius * math.sin(spiral_angle)
        
        if self.dim == 2:
            return {"x": x, "y": y}
        elif self.dim == 3:
            z = spiral_angle / (2 * math.pi)  # Height encodes angle
            return {"x": x, "y": y, "z": z}
        else:
            return {"x": x, "y": y}


class PolarSystem(CoordinateSystem):
    """Polar coordinates (r, θ)."""
    
    def name(self) -> str:
        return "Polar"
    
    def dimensions(self) -> int:
        return 2
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Already in polar form."""
        return {
            "r": spiral_radius,
            "theta": spiral_angle % (2 * math.pi)
        }


class CylindricalSystem(CoordinateSystem):
    """Cylindrical coordinates (ρ, φ, z)."""
    
    def name(self) -> str:
        return "Cylindrical"
    
    def dimensions(self) -> int:
        return 3
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Cylindrical: r is radial distance, θ becomes φ, add height."""
        return {
            "rho": spiral_radius,
            "phi": spiral_angle % (2 * math.pi),
            "z": spiral_angle / (2 * math.pi)
        }


class SphericalSystem(CoordinateSystem):
    """Spherical coordinates (ρ, θ, φ)."""
    
    def name(self) -> str:
        return "Spherical"
    
    def dimensions(self) -> int:
        return 3
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Map spiral to sphere surface."""
        rho = 1.0 + 0.1 * spiral_radius  # Slight bulge from unit sphere
        theta = spiral_angle % (2 * math.pi)
        phi = math.asin(min(1.0, spiral_radius / 2.0))  # Latitude
        
        return {
            "rho": rho,
            "theta": theta,
            "phi": phi
        }


class ComplexSystem(CoordinateSystem):
    """Complex plane z = x + iy."""
    
    def name(self) -> str:
        return "Complex"
    
    def dimensions(self) -> int:
        return 2
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Complex plane representation."""
        real = spiral_radius * math.cos(spiral_angle)
        imag = spiral_radius * math.sin(spiral_angle)
        
        return {
            "z_real": real,
            "z_imag": imag,
            "z_magnitude": spiral_radius,
            "z_argument": spiral_angle
        }


class ToroidalSystem(CoordinateSystem):
    """Toroidal (donut) coordinates (u, v)."""
    
    def __init__(self, major_radius: float = 5.0, minor_radius: float = 2.0):
        self.R = major_radius
        self.r = minor_radius
    
    def name(self) -> str:
        return f"Toroidal(R={self.R},r={self.r})"
    
    def dimensions(self) -> int:
        return 3
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Map spiral onto torus surface."""
        # u: angle around major circle
        u = spiral_angle % (2 * math.pi)
        # v: angle around minor circle, driven by radius
        v = (spiral_radius * 2 * math.pi) % (2 * math.pi)
        
        # Parametric torus
        x = (self.R + self.r * math.cos(v)) * math.cos(u)
        y = (self.R + self.r * math.cos(v)) * math.sin(u)
        z = self.r * math.sin(v)
        
        return {
            "u": u,
            "v": v,
            "x": x,
            "y": y,
            "z": z
        }


class HyperbolicSystem(CoordinateSystem):
    """Hyperbolic plane (Poincaré disk model)."""
    
    def name(self) -> str:
        return "Hyperbolic_PoincareDisk"
    
    def dimensions(self) -> int:
        return 2
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """
        Map to Poincaré disk model.
        Poincaré disk: |z| < 1 represents entire hyperbolic plane.
        """
        # Compress spiral_radius to stay within disk
        # Use tangent map: r_hyperbolic = tanh(spiral_radius/2)
        r_disk = math.tanh(spiral_radius / 4.0)
        
        x = r_disk * math.cos(spiral_angle)
        y = r_disk * math.sin(spiral_angle)
        
        return {
            "x": x,
            "y": y,
            "distance_from_center": math.sqrt(x**2 + y**2)
        }


class ProjectiveSystem(CoordinateSystem):
    """Projective plane (homogeneous coordinates)."""
    
    def name(self) -> str:
        return "Projective"
    
    def dimensions(self) -> int:
        return 3
    
    def embed(self, spiral_radius: float, spiral_angle: float) -> Dict[str, float]:
        """Homogeneous coordinates [X:Y:Z]."""
        x = spiral_radius * math.cos(spiral_angle)
        y = spiral_radius * math.sin(spiral_angle)
        z = 1.0
        
        return {
            "X": x,
            "Y": y,
            "Z": z,
            "ratio_XZ": x / z if z != 0 else 0,
            "ratio_YZ": y / z if z != 0 else 0
        }


class ManifoldEmbedding:
    """
    Complete manifold embedding system.
    Transposes primes onto any coordinate system.
    """
    
    def __init__(self, primes: List[int]):
        self.primes = primes
        self.gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
        
        # Build intrinsic spiral parameters
        self.spiral_angles = self._compute_spiral_angles()
        self.spiral_radii = self._compute_spiral_radii()
    
    def _compute_spiral_angles(self) -> List[float]:
        """Cumulative angles from gap projections."""
        angles = []
        total = 0.0
        
        for i in range(len(self.gaps)):
            prime = self.primes[i]
            gap = self.gaps[i]
            
            # Gap projection: dθ = (gap / prime) × 2π
            delta_angle = (gap / prime) * (2 * math.pi) if prime > 0 else 0
            total += delta_angle
            angles.append(total)
        
        return angles
    
    def _compute_spiral_radii(self, spiral_type: str = "logarithmic") -> List[float]:
        """Radii from spiral equation."""
        radii = []
        
        for i in range(len(self.gaps)):
            if spiral_type == "logarithmic":
                # r(n) = c * log(n+1)
                r = 0.5 * math.log(i + 2)
            elif spiral_type == "archimedean":
                # r(n) = c * n
                r = 0.1 * (i + 1)
            else:
                r = 0.5 * math.log(i + 2)
            
            radii.append(max(r, 0.1))  # Avoid zero
        
        return radii
    
    def embed_onto_system(
        self, 
        coordinate_system: CoordinateSystem
    ) -> List[ManifoldPoint]:
        """
        Embed all primes onto specified coordinate system.
        
        Args:
            coordinate_system: A CoordinateSystem instance
            
        Returns:
            List of ManifoldPoint objects with coordinates
        """
        manifold_points = []
        
        for i in range(len(self.gaps)):
            prime = self.primes[i]
            gap = self.gaps[i]
            spiral_angle = self.spiral_angles[i]
            spiral_radius = self.spiral_radii[i]
            
            # Embed onto coordinate system
            coords = coordinate_system.embed(spiral_radius, spiral_angle)
            
            point = ManifoldPoint(
                prime=prime,
                index=i,
                gap=gap,
                spiral_angle=spiral_angle,
                spiral_radius=spiral_radius,
                coordinates=coords,
                coordinate_system=coordinate_system.name()
            )
            
            manifold_points.append(point)
        
        return manifold_points
    
    def multi_system_embedding(
        self,
        systems: List[CoordinateSystem]
    ) -> Dict[str, List[ManifoldPoint]]:
        """
        Embed onto multiple coordinate systems simultaneously.
        Shows how primes map across different geometric representations.
        """
        results = {}
        
        for system in systems:
            results[system.name()] = self.embed_onto_system(system)
        
        return results
    
    def export_all_coordinates(self, filepath: str) -> None:
        """Export all coordinate representations to file."""
        systems = [
            CartesianSystem(2),
            CartesianSystem(3),
            PolarSystem(),
            CylindricalSystem(),
            SphericalSystem(),
            ComplexSystem(),
            ToroidalSystem(),
            HyperbolicSystem(),
            ProjectiveSystem()
        ]
        
        embeddings = self.multi_system_embedding(systems)
        
        with open(filepath, 'w') as f:
            f.write("Prime Manifold Coordinates - All Representations\n")
            f.write("=" * 100 + "\n\n")
            
            for system_name, points in embeddings.items():
                f.write(f"\n{system_name}\n")
                f.write("-" * len(system_name) + "\n")
                
                # Get coordinate keys from first point
                if points:
                    coord_keys = sorted(points[0].coordinates.keys())
                    
                    # Header
                    header = "prime\tindex\tgap"
                    for key in coord_keys:
                        header += f"\t{key}"
                    f.write(header + "\n")
                    
                    # Data rows (first 50)
                    for p in points[:min(50, len(points))]:
                        row = f"{p.prime}\t{p.index}\t{p.gap}"
                        for key in coord_keys:
                            val = p.coordinates.get(key, "")
                            if isinstance(val, float):
                                row += f"\t{val:.6f}"
                            else:
                                row += f"\t{val}"
                        f.write(row + "\n")


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate primes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]


def demonstrate_coordinate_transposition():
    """Show primes transposed onto various coordinate systems."""
    
    print("\n" + "="*80)
    print("PRIME MANIFOLD EMBEDDING - COORDINATE TRANSPOSITION")
    print("="*80)
    
    # Generate primes
    primes = sieve_of_eratosthenes(2000)
    print(f"\n[1] Generated {len(primes)} primes")
    
    # Create embedding
    embedding = ManifoldEmbedding(primes)
    print(f"[2] Built manifold embedding")
    
    # Define coordinate systems
    systems = [
        CartesianSystem(2),
        PolarSystem(),
        CartesianSystem(3),
        CylindricalSystem(),
        ComplexSystem(),
        ToroidalSystem(),
        HyperbolicSystem(),
    ]
    
    print(f"\n[3] Embedding onto {len(systems)} coordinate systems:")
    for sys in systems:
        print(f"    - {sys.name()} ({sys.dimensions()}D)")
    
    # Multi-system embedding
    embeddings = embedding.multi_system_embedding(systems)
    
    # Show sample points
    print(f"\n[4] Sample embeddings for prime 97:")
    for system_name, points in embeddings.items():
        # Find prime 97
        for p in points:
            if p.prime == 97:
                print(f"\n    {system_name}:")
                for key, val in sorted(p.coordinates.items()):
                    if isinstance(val, float):
                        print(f"      {key}: {val:.6f}")
                    else:
                        print(f"      {key}: {val}")
                break
    
    # Export all coordinates
    export_file = "prime_manifold_coordinates.txt"
    embedding.export_all_coordinates(export_file)
    print(f"\n[5] Exported full coordinate data to {export_file}")
    
    print("\n" + "="*80)
    print("COORDINATE TRANSPOSITION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_coordinate_transposition()
