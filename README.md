# H3D — 3D Heisenberg Model Simulation with Monte Carlo & Machine Learning

**H3D** is a high-performance C++ simulation framework for the **3D classical Heisenberg model**, based on the **Metropolis Monte Carlo algorithm**, with optional **machine-learning-assisted spin-flip proposals**.  
The project focuses on numerical accuracy, parallel performance, and extensibility toward hybrid ML–MC methods.

---

## Features

- Classical **3D Heisenberg spin model** on a cubic lattice
- **Metropolis Monte Carlo** updates with periodic boundary conditions
- Physically correct **uniform sampling on the unit sphere**
- **OpenMP parallelization** over temperature points
- Computation of:
  - Energy ⟨E⟩
  - Magnetization ⟨|M|⟩
  - Magnetic susceptibility χ
  - Specific heat Cᵥ
- Optional **ML-assisted spin flip acceptance** using a TorchScript model
- Data export for post-processing and visualization in Python

---

## How to Use This Repository (Complete File Guide)

This section explains **what every file does**, **when you should use it**, and **how the full workflow fits together** — from simulation, to ML training, to plotting.

---

## 🔧 Core Physics & Monte Carlo Implementation (C++)

### `Metropolis.h`
**What it is**
- Header file declaring the `Heisenberg_Metropolis` class.

**What it defines**
- Physical parameters:
  - lattice size `L`
  - coupling constant `J`
- Monte Carlo parameters:
  - `Ntherm` (thermalization sweeps)
  - `Nsample` (measurement sweeps)
  - `Nsubsweep` (spin updates per sweep)
- Core physics methods:
  - `initialize_lattice()`
  - `step()` (standard Metropolis update)
  - `local_energy()`, `total_energy()`
  - `total_magnetization()`
  - `binning_analysis()` for error estimation

**When to edit**
- Changing the physical model
- Adding new observables
- Extending Monte Carlo algorithms

---

### `Metropolis.cpp`
**What it is**
- Implementation of all methods declared in `Metropolis.h`.

**What happens here**
- Periodic boundary conditions
- Uniform random sampling of spins on the unit sphere
- Metropolis acceptance rule
- Energy and magnetization calculations
- (Optional) ML-assisted Monte Carlo steps:
  - `step_ml()`
  - `step_ml_batch()` (batched inference)

**When to edit**
- Optimizing performance
- Modifying Monte Carlo logic
- Experimenting with ML-accelerated sampling

---

### `helpers.h`
**What it is**
- Lightweight utility header.

**Contains**
- `Vec3D` type (`std::vector<double>`)
- Dot product
- Operator overloading (`+`, `-`, `/`)

---

### `helpers.cpp`
**What it is**
- Implementation of helper math utilities.

**When to edit**
- Rarely — only if extending vector operations

---

## Executable Programs

### `main_parallel.cpp` **Main Simulation Program**
**What it does**
- Runs the full 3D Heisenberg Monte Carlo simulation
- Parallelizes over temperatures using **OpenMP**
- Performs:
  - Thermalization
  - Sampling
  - Observable calculation
- Computes:
  - ⟨E⟩ energy
  - ⟨|M|⟩ magnetization
  - χ magnetic susceptibility
  - Cᵥ specific heat
- Writes results to a CSV file



