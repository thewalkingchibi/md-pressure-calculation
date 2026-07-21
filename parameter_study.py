import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Avogadro, R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    calculate_force,
    ideal_gas_pressure,
    initialize_velocities,
    simulate_NVT_step,
)


# --------------------------------------------------
# FAST STUDY SETTINGS
# --------------------------------------------------
N_PARTICLES = 200
N_STEPS = 1000
DT = 0.01                 # ps
TEMPERATURE = 300.0       # K
TAU_THERMOSTAT = 1.0      # ps
RIJ_MIN = 1e-2            # nm
MASS_ARGON = 39.95

SIGMA_REFERENCE = 0.34
EPSILON_REFERENCE = 120.0 * R * 1e-3
BOX_REFERENCE = 4.0       # nm

SEED = 12345


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
def initialize_lattice(
    ps: ParticleSystem,
    box_length: float,
    jitter_fraction: float = 0.03,
) -> None:
    """
    Place particles on a cubic lattice with a small random displacement.

    This avoids particles starting almost on top of one another in the
    smaller simulation boxes.
    """
    n_side = int(np.ceil(ps.n ** (1.0 / 3.0)))
    spacing = box_length / n_side

    grid = np.indices((n_side, n_side, n_side))
    grid = grid.reshape(3, -1).T[:ps.n]

    positions = (grid + 0.5) * spacing

    jitter = np.random.uniform(
        -jitter_fraction * spacing,
        jitter_fraction * spacing,
        size=positions.shape,
    )

    ps.position[:] = np.mod(
        positions + jitter,
        box_length,
    )


def virial_pressure_from_value(
    virial_kj_per_mol: float,
    box_length: float,
) -> float:
    """
    Convert the virial W returned by calculate_force() from kJ/mol
    into the virial pressure contribution in Pa.
    """
    volume_m3 = box_length**3 * 1e-27

    virial_joule = (
        virial_kj_per_mol
        * 1000.0
        / Avogadro
    )

    return virial_joule / (3.0 * volume_m3)


# --------------------------------------------------
# RUN ONE SIMULATION
# --------------------------------------------------
def run_case(
    sigma: float,
    epsilon: float,
    box_length: float,
    label: str,
) -> dict:
    """
    Run one short NVT simulation and return pressure averages.

    Only the second half of the simulation is used for averaging.
    """
    print(f"Running {label}...")

    # Same random sequence for every parameter set.
    # This makes comparisons more controlled.
    np.random.seed(SEED)

    sim = SimulationParameters(
        dt=DT,
        n_steps=N_STEPS,
        temperature=TEMPERATURE,
        box_length=box_length,
        tau_thermostat=TAU_THERMOSTAT,
        rij_min=RIJ_MIN,
    )

    ps = ParticleSystem(N_PARTICLES)

    ps.mass[:] = MASS_ARGON
    ps.sigma[:] = sigma
    ps.epsilon[:] = epsilon

    initialize_lattice(ps, box_length)
    initialize_velocities(ps, TEMPERATURE)

    # Initial force
    calculate_force(ps, sim)

    ideal_pressures = np.zeros(N_STEPS)
    virial_pressures = np.zeros(N_STEPS)
    total_pressures = np.zeros(N_STEPS)

    for step in range(N_STEPS):
        simulate_NVT_step(ps, sim)

        ideal_pressures[step] = ideal_gas_pressure(
            ps,
            sim,
        )

        # calculate_force() returns the virial W in kJ/mol
        virial_kj_per_mol = calculate_force(ps, sim)

        virial_pressures[step] = (
            virial_pressure_from_value(
                virial_kj_per_mol,
                box_length,
            )
        )

        total_pressures[step] = (
            ideal_pressures[step]
            + virial_pressures[step]
        )

    # Discard the first half as equilibration
    averaging_start = N_STEPS // 2

    result = {
        "label": label,
        "sigma_nm": sigma,
        "epsilon_kj_mol": epsilon,
        "box_length_nm": box_length,
        "volume_nm3": box_length**3,
        "mean_ideal_pa": np.mean(
            ideal_pressures[averaging_start:]
        ),
        "mean_virial_pa": np.mean(
            virial_pressures[averaging_start:]
        ),
        "mean_total_pa": np.mean(
            total_pressures[averaging_start:]
        ),
        "std_total_pa": np.std(
            total_pressures[averaging_start:]
        ),
    }

    print(
        f"  total = "
        f"{result['mean_total_pa'] / 1e6:.4f} MPa, "
        f"virial = "
        f"{result['mean_virial_pa'] / 1e6:.4f} MPa"
    )

    return result


# Avoid repeating identical simulations
simulation_cache = {}


def get_case(
    sigma: float,
    epsilon: float,
    box_length: float,
    label: str,
) -> dict:
    key = (
        round(sigma, 8),
        round(epsilon, 12),
        round(box_length, 8),
    )

    if key not in simulation_cache:
        simulation_cache[key] = run_case(
            sigma=sigma,
            epsilon=epsilon,
            box_length=box_length,
            label=label,
        )

    result = simulation_cache[key].copy()
    result["label"] = label

    return result


# --------------------------------------------------
# 1. EPSILON ANALYSIS
# --------------------------------------------------
epsilon_factors = [
    0.0,
    1.0,
    2.0,
]

epsilon_results = []

for factor in epsilon_factors:
    result = get_case(
        sigma=SIGMA_REFERENCE,
        epsilon=factor * EPSILON_REFERENCE,
        box_length=BOX_REFERENCE,
        label=f"epsilon_{factor:.1f}",
    )

    epsilon_results.append(result)


# --------------------------------------------------
# 2. SIGMA ANALYSIS
# --------------------------------------------------
sigma_values = [
    0.30,
    0.34,
    0.40,
]

sigma_results = []

for sigma in sigma_values:
    result = get_case(
        sigma=sigma,
        epsilon=EPSILON_REFERENCE,
        box_length=BOX_REFERENCE,
        label=f"sigma_{sigma:.2f}",
    )

    sigma_results.append(result)


# --------------------------------------------------
# 3. BOYLE'S LAW
#
# epsilon = 0 removes the LJ interaction.
# The system should then behave as an ideal gas.
# --------------------------------------------------
box_lengths = [
    4.0,
    5.0,
    6.0,
]

boyle_results = []

for box_length in box_lengths:
    result = get_case(
        sigma=SIGMA_REFERENCE,
        epsilon=0.0,
        box_length=box_length,
        label=f"Boyle_L{box_length:.1f}",
    )

    boyle_results.append(result)


# --------------------------------------------------
# SAVE NUMERICAL RESULTS
# --------------------------------------------------
all_results = (
    epsilon_results
    + sigma_results
    + boyle_results
)

with open(
    "parameter_study_results.csv",
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=all_results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# EPSILON PLOT
# --------------------------------------------------
plt.figure(figsize=(8, 6))

plt.plot(
    epsilon_factors,
    [
        result["mean_total_pa"] / 1e6
        for result in epsilon_results
    ],
    marker="o",
    label="mean total pressure",
)

plt.plot(
    epsilon_factors,
    [
        result["mean_virial_pa"] / 1e6
        for result in epsilon_results
    ],
    marker="o",
    label="mean virial contribution",
)

plt.xlabel(
    r"$\epsilon / \epsilon_{\mathrm{Ar}}$",
    fontsize=14,
)

plt.ylabel(
    "mean pressure [MPa]",
    fontsize=14,
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "epsilon_pressure_comparison.png",
    dpi=300,
)

plt.show()


# --------------------------------------------------
# SIGMA PLOT
# --------------------------------------------------
plt.figure(figsize=(8, 6))

plt.plot(
    sigma_values,
    [
        result["mean_total_pa"] / 1e6
        for result in sigma_results
    ],
    marker="o",
    label="mean total pressure",
)

plt.plot(
    sigma_values,
    [
        result["mean_virial_pa"] / 1e6
        for result in sigma_results
    ],
    marker="o",
    label="mean virial contribution",
)

plt.xlabel(
    r"$\sigma$ [nm]",
    fontsize=14,
)

plt.ylabel(
    "mean pressure [MPa]",
    fontsize=14,
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "sigma_pressure_comparison.png",
    dpi=300,
)

plt.show()


# --------------------------------------------------
# BOYLE PLOT: P AGAINST 1/V
# --------------------------------------------------
volumes_nm3 = np.array([
    result["volume_nm3"]
    for result in boyle_results
])

inverse_volumes = 1.0 / volumes_nm3

pressures_pa = np.array([
    result["mean_total_pa"]
    for result in boyle_results
])

linear_fit = np.polyfit(
    inverse_volumes,
    pressures_pa,
    1,
)

fitted_pressures = np.polyval(
    linear_fit,
    inverse_volumes,
)

plt.figure(figsize=(8, 6))

plt.scatter(
    inverse_volumes,
    pressures_pa / 1e6,
    label="simulation",
)

plt.plot(
    inverse_volumes,
    fitted_pressures / 1e6,
    label="linear fit",
)

plt.xlabel(
    r"$1/V$ [nm$^{-3}$]",
    fontsize=14,
)

plt.ylabel(
    "mean total pressure [MPa]",
    fontsize=14,
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "boyle_law.png",
    dpi=300,
)

plt.show()


# --------------------------------------------------
# NUMERICAL BOYLE CHECK: P * V
# --------------------------------------------------
print("")
print("Boyle-law check")
print("----------------")

pv_values = []

for result in boyle_results:
    volume_m3 = (
        result["volume_nm3"]
        * 1e-27
    )

    pv_joule = (
        result["mean_total_pa"]
        * volume_m3
    )

    pv_values.append(pv_joule)

    print(
        f"L = {result['box_length_nm']:.1f} nm: "
        f"P*V = {pv_joule:.6e} J"
    )

relative_spread = (
    (max(pv_values) - min(pv_values))
    / np.mean(pv_values)
    * 100.0
)

print(
    f"Relative spread in P*V: "
    f"{relative_spread:.2f} %"
)
