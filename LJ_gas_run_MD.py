#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_run_MD.py

Main program for running molecular dynamics simulations using Lennard-Jones particles.
Initializes the system, runs the integrator loop, records energy and trajectory data, 
and visualizes results.

Author: Bettina Keller
Created: May 28, 2025

This script imports all classes and functions from md_simulation.py and controls
the simulation workflow.

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt

import time
from datetime import datetime

from LJ_gas import(
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    calculate_force,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure,
    virial_pressure,
    total_pressure
    )

#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------
# Define tic and toc functions
def tic():
    """Start a timer."""
    global _tic_time
    _tic_time = time.time()

def toc():
    """Stop the timer and return the elapsed time in seconds."""

    elapsed_time = None
    
    if '_tic_time' in globals():
        elapsed_time = time.time() - _tic_time
    
    else:
        print("Error: tic() was not called before toc()")
    
    return elapsed_time

def run_quick_simulation(sigma, epsilon, mass, n_particles=100, n_steps=300,
                          dt=0.1, temperature=300, box_length=100,
                          tau_thermostat=1, rij_min=1e-2, n_equil=100,
                          seed=None):
    """
    Runs a short, standalone NVT simulation for a given (sigma, epsilon) pair
    and returns the time-averaged ideal and virial pressures.
 
    This is used to scan how sigma and epsilon affect the pressure, without
    disturbing the main simulation's ParticleSystem/SimulationParameters
    objects or trajectories.
 
    Parameters:
        sigma, epsilon: Lennard-Jones parameters to test (nm, kJ/mol).
        mass: particle mass in u (kept fixed across the sweep).
        n_particles, n_steps, dt, temperature, box_length, tau_thermostat,
        rij_min: simulation settings for the short run (smaller/shorter than
                 the main production run, purely to keep the sweep fast).
        n_equil: number of initial steps to discard as equilibration before
                 averaging the pressures.
        seed: optional RNG seed for reproducibility.
 
    Returns:
        (P_ideal_avg, P_virial_avg) in Pascals.
    """
    if seed is not None:
        np.random.seed(seed)
 
    sim_q = SimulationParameters(dt=dt, n_steps=n_steps, temperature=temperature,
                                  box_length=box_length,
                                  tau_thermostat=tau_thermostat, rij_min=rij_min)
 
    ps_q = ParticleSystem(n_particles)
    for i in range(n_particles):
        ps_q.set_parameters(i, mass=mass, sigma=sigma, epsilon=epsilon)
 
    initialize_positions(ps_q, sim_q.box_length)
    initialize_velocities(ps_q, sim_q.temperature)
    calculate_force(ps_q, sim_q)
 
    P_ideal_series = np.zeros(n_steps + 1)
    P_virial_series = np.zeros(n_steps + 1)
    P_ideal_series[0] = ideal_gas_pressure(ps_q, sim_q)
    P_virial_series[0] = total_pressure(ps_q, sim_q)
 
    for i in range(n_steps):
        simulate_NVT_step(ps_q, sim_q)
        P_ideal_series[i + 1] = ideal_gas_pressure(ps_q, sim_q)
        P_virial_series[i + 1] = total_pressure(ps_q, sim_q)
 
    # average over the post-equilibration window
    P_ideal_avg = np.mean(P_ideal_series[n_equil:])
    P_virial_avg = np.mean(P_virial_series[n_equil:])
 
    return P_ideal_avg, P_virial_avg
 

#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.1             # ps
n_steps = 1000 
temperature = 300     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = True          # switch to decide between NVT and NVE

# output
file_name_base = "my_simulation"  # file name for all output files

#----------------------------------------------------------------
#   P R O G R A M
#----------------------------------------------------------------
# start the timer
tic()

#
# initialize simulation parameters
#
sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps, 
                           temperature = temperature, 
                           box_length = box_length, 
                           tau_thermostat = tau_thermostat,
                           rij_min=rij_min
                           )

#
# initialize ParticleSystem 
#
ps = ParticleSystem(n_particles)

# fill in the parameters for argon
for i in range(n_particles): 
    ps.set_parameters(i, mass=mass_argon, sigma=sigma_argon, epsilon=epsilon_argon)

# set initial positions     
initialize_positions(ps, sim.box_length)

# set initial velocities     
initialize_velocities(ps, sim.temperature)

# calculate force according to initial positions
calculate_force(ps, sim)

# calculate box density
rho = density(ps, sim)

# calculate initial values of variable properties
E_pot_init = potential_energy(ps, sim)
E_kin_init = kinetic_energy(ps)
T_init = instantaneous_temperature(ps)
P_init = ideal_gas_pressure(ps, sim)


# initialize position trajectory
position_trajectory = np.zeros((sim.n_steps+1, n_particles, 3))
position_trajectory[0,:,:] = ps.position # initial position

# initialize energy trajectory
energy_trajectory = np.zeros((sim.n_steps+1, 6))
energy_trajectory[0,0] = potential_energy( ps, sim)       # potential energy
energy_trajectory[0,1] = kinetic_energy(ps)               # kinetic energy
energy_trajectory[0,2] = instantaneous_temperature(ps)    # instantaneous pressure
energy_trajectory[0,3] = ideal_gas_pressure(ps, sim)      # ideal gas pressure
energy_trajectory[0,4] = virial_pressure(ps, sim)         # virial contribution
energy_trajectory[0,5] = energy_trajectory[0,3] + energy_trajectory[0,4]    # total pressure (ideal + virial)


#--------------------------------------------------
#  The acutal MD simulation
#--------------------------------------------------
for i in range(sim.n_steps):
    if NVT==True:
        simulate_NVT_step(ps, sim)
    else: 
        simulate_NVE_step(ps, sim)
        
    # store updated positions
    position_trajectory[i+1,:,:] = ps.position # store updated positions

    # store updated energies, temperature and pressure
    energy_trajectory[i+1,0] = potential_energy( ps, sim)     # potential energy
    energy_trajectory[i+1,1] = kinetic_energy(ps)             # kinetic energy
    energy_trajectory[i+1,2] = instantaneous_temperature(ps)  # instantaneous pressure
    energy_trajectory[i+1,3] = ideal_gas_pressure(ps, sim)    # ideal gas pressure
    energy_trajectory[i+1,4] = virial_pressure(ps, sim)       # virial contribution
    energy_trajectory[i+1,5] = total_pressure(ps, sim)        # total pressure

#--------------------------------------
# W R I T E    T R A J E C T O R I E S 
#--------------------------------------
# write position trajectory to file
write_xyz_trajectory(file_name_base + "_pos.xyz", position_trajectory, atom_symbol="Ar")
# write energy trajectory to file (binary and text)
np.save(file_name_base + "_ene.npy", energy_trajectory)
np.savetxt(file_name_base + "_ene.dat", energy_trajectory, fmt="%.6e", header="#E_pot  E_kin  T  P_ideal P_virial", comments='')


#----------------------------------------------------
# P L O T   E N E R G Y   T R A J E C T O R I E S
#----------------------------------------------------
# set time axis
time_ps = np.arange(sim.n_steps + 1) * sim.dt

#
# potential energy
# 
E_pot_min = np.mean(energy_trajectory[:,0]) - 1   # lower limit of E_pot axis
E_pot_max = np.mean(energy_trajectory[:,0]) + 1   # upper limit of E_pot axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,0]) 
plt.ylim(E_pot_min, E_pot_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_pot [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Epot.png", dpi=300, bbox_inches='tight')
plt.show()

#
# kinetic energy
# 
E_kin_min = np.mean(energy_trajectory[:,1]) - 100   # lower limit of E_kin axis
E_kin_max = np.mean(energy_trajectory[:,1]) + 100   # upper limit of E_kin axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,1]) 
plt.ylim(E_kin_min, E_kin_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_kin [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Ekin.png", dpi=300, bbox_inches='tight')
plt.show()

#
# temperature
# 
T_min = np.mean(energy_trajectory[:,2]) - 100   # lower limit of T axis
T_max = np.mean(energy_trajectory[:,2]) + 100   # upper limit of T axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,2]) 
plt.ylim(T_min, T_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("T [K]", fontsize=14)

plt.savefig(file_name_base + "_T.png", dpi=300, bbox_inches='tight')
plt.show()

#
# ideal gas pressure
# 
Pideal_min = np.mean(energy_trajectory[:,3]) - 200   # lower limit of P axis
Pideal_max = np.mean(energy_trajectory[:,3]) + 200   # upper limit of P axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,3]) 
plt.ylim(Pideal_min, Pideal_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("P_ideal [Pa]", fontsize=14)

plt.savefig(file_name_base + "_Pideal.png", dpi=300, bbox_inches='tight')
plt.show()

#
# total pressure 
#
Ptotal_min = np.mean(energy_trajectory[:,5]) - 200   # lower limit of P axis
Ptotal_max = np.mean(energy_trajectory[:,5]) + 200   # upper limit of P axis
 
plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,5])
plt.ylim(Ptotal_min, Ptotal_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("P_total [Pa]", fontsize=14)
 
plt.savefig(file_name_base + "_Pvirial.png", dpi=300, bbox_inches='tight')
plt.show()
 
#
# ideal pressure + total pressure + virial contribution
#
plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,3], label="ideal gas pressure", alpha=0.8)
plt.plot(time_ps, energy_trajectory[:,5], label="total pressure", alpha=0.8)
plt.plot(time_ps, energy_trajectory[:,4], label="virial contribution", alpha=0.8)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("P [Pa]", fontsize=14)
plt.legend(fontsize=12)
plt.title("Ideal-gas vs. virial pressure")
 
plt.savefig(file_name_base + "_P_compare.png", dpi=300, bbox_inches='tight')
plt.show()
 
 
#----------------------------------------------------
# E F F E C T   O F   S I G M A   A N D   E P S I L O N   O N   P R E S S U R E
#----------------------------------------------------
# The ideal-gas pressure only depends on N, T and V, so it is completely
# insensitive to sigma and epsilon - only the virial pressure "feels" the
# Lennard-Jones interactions. These short, independent sweep simulations
# make that visible directly.
#
# Sweeps use a smaller system / fewer steps than the production run above,
# purely to keep the total runtime reasonable; increase n_particles/n_steps
# for smoother, more accurate averages.
 
print("\nRunning sigma sweep (this may take a while)...")
 
sigma_values = np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.40])   # nm
epsilon_fixed = epsilon_argon
 
P_ideal_vs_sigma = np.zeros_like(sigma_values)
P_virial_vs_sigma = np.zeros_like(sigma_values)
 
for idx, s in enumerate(sigma_values):
    P_ideal_vs_sigma[idx], P_virial_vs_sigma[idx] = run_quick_simulation(
        sigma=s, epsilon=epsilon_fixed, mass=mass_argon, seed=0)
 
plt.figure(figsize=(8, 6))
plt.plot(sigma_values, P_ideal_vs_sigma, 'o-', label="ideal gas pressure")
plt.plot(sigma_values, P_virial_vs_sigma, 's-', label="virial pressure")
plt.xlabel("sigma [nm]", fontsize=14)
plt.ylabel("time-averaged P [Pa]", fontsize=14)
plt.legend(fontsize=12)
plt.title(f"Pressure vs. sigma (epsilon = {epsilon_fixed:.4f} kJ/mol)")
 
plt.savefig(file_name_base + "_P_vs_sigma.png", dpi=300, bbox_inches='tight')
plt.show()
 
print("Running epsilon sweep (this may take a while)...")
 
epsilon_values = np.linspace(0.5, 2.0, 6) * epsilon_argon
sigma_fixed = sigma_argon
 
P_ideal_vs_epsilon = np.zeros_like(epsilon_values)
P_virial_vs_epsilon = np.zeros_like(epsilon_values)
 
for idx, e in enumerate(epsilon_values):
    P_ideal_vs_epsilon[idx], P_virial_vs_epsilon[idx] = run_quick_simulation(
        sigma=sigma_fixed, epsilon=e, mass=mass_argon, seed=0)
 
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, P_ideal_vs_epsilon, 'o-', label="ideal gas pressure")
plt.plot(epsilon_values, P_virial_vs_epsilon, 's-', label="virial pressure")
plt.xlabel("epsilon [kJ/mol]", fontsize=14)
plt.ylabel("time-averaged P [Pa]", fontsize=14)
plt.legend(fontsize=12)
plt.title(f"Pressure vs. epsilon (sigma = {sigma_fixed:.3f} nm)")
 
plt.savefig(file_name_base + "_P_vs_epsilon.png", dpi=300, bbox_inches='tight')
plt.show()


#--------------------------------------
# O U T P U T 
#--------------------------------------
elapsed_time = toc()   # stop the timer
output_lines = []

output_lines.append("")
output_lines.append("----------------------------------------------------------")
output_lines.append("Simulation parameters ")    
output_lines.append("----------------------------------------------------------")
output_lines.append(f"{'Number of particles:':<30}{ps.n:>10.0f} ")
output_lines.append(f"{'Box length:':<30}{sim.box_length:>10.3e} nm")
output_lines.append(f"{'Box volume:':<30}{sim.box_length**3:>10.3e} nm^3")
output_lines.append(f"{'Density:':<30}{rho:>10.3e} g/cm^3")
output_lines.append("")   
output_lines.append(f"{'Time step:':<30}{sim.dt:>10.3f} ps")
output_lines.append(f"{'Number of time steps:':<30}{sim.n_steps:>10.0f}")
output_lines.append(f"{'Simulation time:':<30}{sim.n_steps * sim.dt :>10.3e} ps")
output_lines.append("")   
if NVT==True: 
    output_lines.append(f"{'Ensemble:':<30}{'NVT':>10}")
    output_lines.append(f"{'Thermostat temperature:':<30}{sim.temperature:>10.0f} K")
    output_lines.append(f"{'Thermostat coupling:':<30}{sim.tau_thermostat:>10.3e} ps")
else: 
    output_lines.append(f"{'Ensemble:':<30}{'NVE':>10}")
    output_lines.append(f"{'Initial velocities:':<30}{sim.temperature:>10.0f} K")

output_lines.append("")     
output_lines.append(f"{'Lower cutoff radius:':<30}{sim.rij_min:>10.3f} nm")
output_lines.append("----------------------------------------------------------")
if elapsed_time: 
    time_per_time_step = elapsed_time/sim.n_steps
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append(f"{'Elapsed time:':<30}{elapsed_time:>10.3f} s")   
    output_lines.append(f"{'Elapsed time per time step:':<30}{time_per_time_step:>10.3f} s")
    output_lines.append(f"{'Time stamp:':<30}{now} s")
output_lines.append("----------------------------------------------------------")
output_lines.append("END")  
output_lines.append("----------------------------------------------------------")

# Print to screen
for line in output_lines:
    print(line)
  
# Write to file
with open(file_name_base + ".out", "w") as f:
    for line in output_lines:
        f.write(line + "\n")    