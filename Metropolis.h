#ifndef HEISENBERG_METROPOLIS_H
#define HEISENBERG_METROPOLIS_H

#include <vector>
#include <random>
#include <tuple>
#include <torch/torch.h>
#include <torch/script.h>

typedef std::vector<double> Vec3D;

class Heisenberg_Metropolis {
public:
    // Constructors
    Heisenberg_Metropolis(int lattice_size);
    Heisenberg_Metropolis(int lattice_size, std::mt19937& generator);

    // binning analysis
    std::vector<double> binning_analysis(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, int Nsamples, int max_l);

    // Random vector and lattice generation
    Vec3D random_unit_vector();
    std::vector<std::vector<std::vector<Vec3D>>> initialize_lattice(const int& L);

    // Functions to calculate energy and magentization
    double energy_average(const std::vector<double>& E);
    Vec3D magnetization_average(const std::vector<Vec3D>& M);
    double local_energy(const std::vector<std::vector<std::vector<Vec3D>>>& lattice,
                        int i, int j, int k);
    double total_energy(const std::vector<std::vector<std::vector<Vec3D>>>& lattice);
    std::vector<double> total_magnetization(
        const std::vector<std::vector<std::vector<Vec3D>>>& lattice);

    
    // step functions
    void step(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp);
    void step_ml(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, torch::jit::script::Module& model);
    void step_ml_batch(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, torch::jit::script::Module& model, int& batchsize);

    // public members
    const double J;
    int L, Ntherm, Nsample, Nsubsweep;

    std::mt19937 gen;
    std::uniform_int_distribution<int> dist;
    std::uniform_real_distribution<double> dist_angle, dist_theta, dist_accept;
};

#endif // HEISENBERG_METROPOLIS_H
