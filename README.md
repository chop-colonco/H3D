# H3D — 3D Heisenberg Model Simulation with Monte Carlo & Machine Learning

**H3D** is a high-performance C++ simulation framework for the **3D classical Heisenberg model**, based on the **Metropolis Monte Carlo algorithm**, with optional **machine-learning-assisted spin-flip proposals**.  
The project focuses on numerical accuracy, parallel performance, and extensibility toward hybrid ML–MC methods.

---

## 📌 Features

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

## 📁 Project Structure

```text
.
├── CMakeLists.txt          # Build configuration
├── Metropolis.h / .cpp     # Heisenberg Metropolis class
├── helpers.h / .cpp        # Vector algebra & utilities
├── main_parallel.cpp       # Parallel Monte Carlo simulation
├── collect_data.cpp        # (Optional) data collection utilities
├── train_model_v2.py       # Train ML spin-flip model (PyTorch)
├── spinflip_model_v2.pt    # Trained TorchScript model
├── Plotting_Data.ipynb     # Analysis & visualization notebook
├── README.md               # Project documentation
└── .gitignore
