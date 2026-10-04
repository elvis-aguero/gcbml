"""Empirical benchmark problems for gcbml (NumPy/SciPy only). See benchmarks/README.md."""

from benchmarks.base import BenchmarkProblem
from benchmarks.problems.b1_poisson import Poisson2D
from benchmarks.problems.b2_upwind import UpwindBoundaryLayer
from benchmarks.problems.b3_mixing import MixingTime
from benchmarks.problems.b4_sde import OrnsteinUhlenbeckEM
from benchmarks.problems.b5_kink import MidpointOnJump
from benchmarks.problems.b6_heat import HeatTwoResolutions
from benchmarks.problems.b7_thin_layer import ThinLayerTrap

ALL_PROBLEMS: dict[str, type[BenchmarkProblem]] = {
    cls.name: cls
    for cls in (
        Poisson2D,
        UpwindBoundaryLayer,
        MixingTime,
        OrnsteinUhlenbeckEM,
        MidpointOnJump,
        HeatTwoResolutions,
        ThinLayerTrap,
    )
}
