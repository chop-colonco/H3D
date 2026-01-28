#include <torch/torch.h>
#include <torch/script.h>
#include "Metropolis.h"
#include "helpers.h"
#include <cmath>
#include <ctime>
#include <iostream>
#include <unordered_set>
#include <tuple>
#include <cassert>

// Constructor
Heisenberg_Metropolis::Heisenberg_Metropolis(int lattice_size)
    : J(1.0), L(lattice_size),
      Ntherm(1e4),
      Nsample(1000),
      Nsubsweep(L * L),
      gen(static_cast<unsigned>(std::time(0))),
      dist(0, L - 1),
      dist_accept(0.0, 1.0) {}

Heisenberg_Metropolis::Heisenberg_Metropolis(int lattice_size, std::mt19937& generator)
    : J(1.0), L(lattice_size),
    Ntherm(1e4),
    Nsample(1000),
    Nsubsweep(L * L),
    gen(generator),
    dist(0, L - 1),
    dist_accept(0.0, 1.0) {}


// Methods:
std::vector<double> Heisenberg_Metropolis::binning_analysis(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, int Nsamples, int max_l) {
        
        assert(std::log2(Nsamples) == std::floor(std::log2(Nsamples)) && "Nsamples must be a power of two");
        assert((1<<max_l)+1 <= Nsamples && "2**l + 1 must be smaller than or equal to Nsamples");

        std::vector<double> M(Nsamples, 0.0);
        std::vector<double> M_vec;
        
        for(int i = 0; i < M.size(); ++i) {
            step(lattice, temp);
            M_vec = total_magnetization(lattice)  / (L * L * L);
            M.at(i) = std::sqrt(dot(M_vec, M_vec));
        } 

        std::vector<double> Delta_l = {};

        for(int l = 0; l < max_l+1; ++l) {

            int N_current = Nsamples/(1<<l);
            double M_mean = std::accumulate(M.begin(), M.end(), 0.0) / M.size();
            std::vector<double> M_minus_mean_squared(M.size(), 0.0);

            for(int i = 0; i < M.size(); ++i) {
                M_minus_mean_squared.at(i) = (M.at(i) - M_mean)*(M.at(i) - M_mean);
            }

            Delta_l.push_back(std::sqrt(1/(static_cast<double>(N_current * static_cast<double>(N_current-1))) * std::accumulate(M_minus_mean_squared.begin(), M_minus_mean_squared.end(), 0.0)));
            
            std::vector<double> M_next;
            for(int i = 0; i< M.size()-1; i += 2) {
                M_next.push_back((M.at(i) + M.at(i+1))/2);
            }
            M = M_next;        
        }
        return Delta_l;     
    }

Vec3D Heisenberg_Metropolis::random_unit_vector() {
    std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * 3.1415926535);
    std::uniform_real_distribution<double> dist_u(-1.0, 1.0);

    double theta = acos(dist_u(gen));
    double phi = dist_phi(gen);

    double x = sin(theta) * cos(phi);
    double y = sin(theta) * sin(phi);
    double z = cos(theta);
    return {x, y, z};
}


std::vector<std::vector<std::vector<Vec3D>>> Heisenberg_Metropolis::initialize_lattice(const int& L) {

    std::vector<std::vector<std::vector<Vec3D>>> lattice(
        L, std::vector<std::vector<Vec3D>>(
            L, std::vector<Vec3D>(L)
        )
    );
    for (int i = 0; i < L; ++i){
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k){
                lattice[i][j][k] = random_unit_vector();
            }
        }
    }
    return lattice;
}

double Heisenberg_Metropolis::energy_average(const std::vector<double>& E) {
    double sum = 0.0;
    for (double e : E) sum += e;
    return sum / E.size();
}

Vec3D Heisenberg_Metropolis::magnetization_average(const std::vector<Vec3D>& M) {
    Vec3D avg = {0.0, 0.0, 0.0};
    for (const Vec3D& m : M) {
        avg[0] += m[0];
        avg[1] += m[1];
        avg[2] += m[2];
    }
    avg[0] /= M.size();
    avg[1] /= M.size();
    avg[2] /= M.size();
    return avg;
}

double Heisenberg_Metropolis::local_energy(const std::vector<std::vector<std::vector<Vec3D>>>& lattice, int i, int j, int k) {
    double energy = 0.0;

    Vec3D seed = lattice[i][j][k];
    Vec3D neighbour_sum = 
        lattice[(i + L + 1) % L][j][k] +
        lattice[(i + L - 1) % L][j][k] +
        lattice[i][(j + L + 1) % L][k] +
        lattice[i][(j + L - 1) % L][k] +
        lattice[i][j][(k + L + 1) % L] +
        lattice[i][j][(k + L - 1) % L];

    return -J * dot(seed, neighbour_sum);
}

double Heisenberg_Metropolis::total_energy(const std::vector<std::vector<std::vector<Vec3D>>>& lattice) {
    double energy = 0.0;
    for (size_t n = 0; n < lattice.size(); ++n) {
        for (size_t m = 0; m < lattice[0].size(); ++m) {
            for (size_t l = 0; l < lattice[0][0].size(); ++l) {
                energy += local_energy(lattice, n, m, l);
            }
        }
    }
    return energy * 0.5;
}

std::vector<double> Heisenberg_Metropolis::total_magnetization(const std::vector<std::vector<std::vector<Vec3D>>>& lattice) {
    std::vector<double> M = {0.0, 0.0, 0.0};
    for (size_t n = 0; n < lattice.size(); ++n) {
        for (size_t m = 0; m < lattice[0].size(); ++m) {
            for (size_t l = 0; l < lattice[0][0].size(); ++l) {
                M = M + lattice[n][m][l];
            }
        }
    }
    return M;
}

void Heisenberg_Metropolis::step(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp) {

    int i = dist(gen);
    int j = dist(gen);
    int k = dist(gen);

    Vec3D old_spin = lattice[i][j][k];
    Vec3D new_spin = random_unit_vector();

    double old_energy = local_energy(lattice, i, j, k);
    lattice[i][j][k] = new_spin;
    double new_energy = local_energy(lattice, i, j, k);
    lattice[i][j][k] = old_spin;

    double delta_E = new_energy - old_energy;

    // Analytically calculate acceptance probability
    double R = dist_accept(gen);
    if (delta_E < 0 || R < std::exp(-delta_E / temp)) {
        lattice[i][j][k] = new_spin;
    }
}


void Heisenberg_Metropolis::step_ml(std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, torch::jit::script::Module& model) {

    // Choose random lattice size
    int i = dist(gen);
    int j = dist(gen);
    int k = dist(gen);

    int L = lattice.size();
    double L_min = 2.0, L_max = 20.0;
    double T_min = 0.1, T_max = 5.0;

    double norm_L = (double(L) - L_min) / (L_max - L_min);
    double norm_temp = (temp - T_min) / (T_max - T_min);

    // Save old and new spin
    Vec3D old_spin = lattice[i][j][k];
    Vec3D new_spin = random_unit_vector();

    // Inspect neighbours
    Vec3D spin1 = lattice[(i + L + 1) % L][j][k];
    Vec3D spin2 = lattice[(i + L - 1) % L][j][k];
    Vec3D spin3 = lattice[i][(j + L + 1) % L][k];
    Vec3D spin4 = lattice[i][(j + L - 1) % L][k];
    Vec3D spin5 = lattice[i][j][(k + L + 1) % L];
    Vec3D spin6 = lattice[i][j][(k + L - 1) % L];

    // Create input tensor
    torch::Tensor input = torch::tensor({
        old_spin[0], old_spin[1], old_spin[2],
        new_spin[0], new_spin[1], new_spin[2],
        spin1[0], spin1[1], spin1[2],
        spin2[0], spin2[1], spin2[2],
        spin3[0], spin3[1], spin3[2],
        spin4[0], spin4[1], spin4[2],
        spin5[0], spin5[1], spin5[2],
        spin6[0], spin6[1], spin6[2],
        norm_L,norm_temp}, torch::kDouble).unsqueeze(0);

    // Calculate Acceptance probability
    torch::Tensor y_test_pred = model.forward({input}).toTensor();
    double prob_accept = y_test_pred.item<double>();
    prob_accept = std::min(1.0, std::max(0.0, prob_accept));
    double R = dist_accept(gen);
    if (R < prob_accept) {
        lattice[i][j][k] = new_spin;
    }
}



struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int>& t) const {
        auto [i, j, k] = t;
        return std::hash<int>()(i) ^ (std::hash<int>()(j) << 1) ^ (std::hash<int>()(k) << 2);
    }
};


void Heisenberg_Metropolis::step_ml_batch(
    std::vector<std::vector<std::vector<Vec3D>>>& lattice,
    double temp,
    torch::jit::script::Module& model,
    int& batchsize) {

    int L = lattice.size();
    double L_min = 2.0, L_max = 20.0;
    double T_min = 0.1, T_max = 5.0;

    double norm_L = (double(L) - L_min) / (L_max - L_min);
    double norm_temp = (temp - T_min) / (T_max - T_min);

    // Pre-allocate tensor
    torch::Tensor batch = torch::empty({batchsize, 26}, torch::kDouble);

    // Store new spins and positions
    std::vector<std::tuple<int, int, int, Vec3D>> updates;
    std::unordered_set<std::tuple<int, int, int>, TupleHash> occupied;

    int n = 0;
    while (n < batchsize) {
        int i = dist(gen);
        int j = dist(gen);
        int k = dist(gen);
        std::tuple<int, int, int> ijk = {i, j, k};

        if (occupied.count(ijk)) continue;

        // Reserve spin and neighbors
        occupied.insert({i, j, k});
        occupied.insert({(i + L + 1) % L, j, k});
        occupied.insert({(i + L - 1) % L, j, k});
        occupied.insert({i, (j + L + 1) % L, k});
        occupied.insert({i, (j + L - 1) % L, k});
        occupied.insert({i, j, (k + L + 1) % L});
        occupied.insert({i, j, (k + L - 1) % L});

        // Prepare input data
        Vec3D old_spin = lattice[i][j][k];
        Vec3D new_spin = random_unit_vector();
        Vec3D spin1 = lattice[(i + L + 1) % L][j][k];
        Vec3D spin2 = lattice[(i + L - 1) % L][j][k];
        Vec3D spin3 = lattice[i][(j + L + 1) % L][k];
        Vec3D spin4 = lattice[i][(j + L - 1) % L][k];
        Vec3D spin5 = lattice[i][j][(k + L + 1) % L];
        Vec3D spin6 = lattice[i][j][(k + L - 1) % L];

        // Fill row in batch tensor
        auto row = batch[n];
        row[0] = old_spin[0]; row[1] = old_spin[1]; row[2] = old_spin[2];
        row[3] = new_spin[0]; row[4] = new_spin[1]; row[5] = new_spin[2];
        row[6] = spin1[0]; row[7] = spin1[1]; row[8] = spin1[2];
        row[9] = spin2[0]; row[10] = spin2[1]; row[11] = spin2[2];
        row[12] = spin3[0]; row[13] = spin3[1]; row[14] = spin3[2];
        row[15] = spin4[0]; row[16] = spin4[1]; row[17] = spin4[2];
        row[18] = spin5[0]; row[19] = spin5[1]; row[20] = spin5[2];
        row[21] = spin6[0]; row[22] = spin6[1]; row[23] = spin6[2];
        row[24] = norm_L;
        row[25] = norm_temp;

        updates.push_back({i, j, k, new_spin});
        ++n;
    }

    // Move to GPU if possible
    // if (torch::cuda::is_available()) {
    //     model.to(torch::kCUDA);
    //     batch = batch.to(torch::kCUDA);
    // }

    // Run forard pass
    torch::Tensor probs = model.forward({batch}).toTensor().to(torch::kCPU);

    // Apply spin updates
    for (int n = 0; n < batchsize; ++n) {
        double prob_accept = probs[n].item<double>();
        prob_accept = std::min(1.0, std::max(0.0, prob_accept));
        double R = dist_accept(gen);
        if (R < prob_accept) {
            auto [i, j, k, new_spin] = updates[n];
            lattice[i][j][k] = new_spin;
        }
    }
}
