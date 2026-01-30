#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <tuple>
#include "helpers.h"
#include "Metropolis.h"
#include <torch/torch.h>
#include <torch/script.h>

typedef std::vector<double> Vec3D;

void sweep_batched(Heisenberg_Metropolis& sim, std::vector<std::vector<std::vector<Vec3D>>>& lattice, double temp, torch::jit::script::Module& model, int Nsubsweep, int batchsize) {
    int full_batches = Nsubsweep / batchsize;
    int remainder = Nsubsweep % batchsize;
    for (int m = 0; m < full_batches; ++m)
        sim.step_ml_batch(lattice, temp, model, batchsize);
    if (remainder > 0)
        sim.step_ml_batch(lattice, temp, model, remainder);
}


int main() {

    // Load model
    torch::jit::script::Module model;
    model = torch::jit::load("/mnt/c/Users/aless/OneDrive/Desktop/CSP/H3D/Metropolis/spinflip_model.pt");
    model.eval();

    // Load Class
    Heisenberg_Metropolis HM(5); // --> addjust lattice size

    // Initial Parameters
    const int L = HM.L; // Lattice
    int Ntherm = HM.Ntherm;
    int Nsample = HM.Nsample;
    int Nsubsweep = HM.Nsubsweep;

    // Temperatures
    std::vector<double> Temperatures;
    for (int i = 1; i <= 50; ++i ) {
        Temperatures.push_back(i/20.0);
        if (i == 28) {
            Temperatures.push_back(1.42);
            Temperatures.push_back(1.43);
            Temperatures.push_back(1.44);
        }
        if (i == 29) {
            Temperatures.push_back(1.46);
            Temperatures.push_back(1.47);
            Temperatures.push_back(1.48);
        }
    }
    
    
    

    // Initialize Arrays
    std::vector<double> E_arr(Temperatures.size());
    std::vector<double> M_arr(Temperatures.size());
    std::vector<double> chi_arr(Temperatures.size());
    std::vector<double> cv_arr(Temperatures.size());

    auto start = std::chrono::high_resolution_clock::now();

    // Iterate over temperatures
    #pragma omp parallel for
    for (size_t idx = 0; idx < Temperatures.size(); ++idx) {

        double temp = Temperatures[idx];

        #pragma omp critical 
        {
            std::cout << "Processing for temperature: " << temp 
                    << " in thread " << omp_get_thread_num() << std::endl;
        }

        // std::cout << "Processing for temperature: " << temp << std::endl;
        
        Heisenberg_Metropolis HM_local(L);

        // Initialize vectors to dave values
        std::vector<double> Energies;
        std::vector<double> Energies2;
        std::vector<double> Mags;
        std::vector<double> Mags2;
        
        // Initialize lattice with all spins aligned
        // std::vector<std::vector<std::vector<Vec3D>>> lattice(
        //     L, std::vector<std::vector<Vec3D>>(
        //         L, std::vector<Vec3D>(
        //             L, Vec3D{0, 0, 1.0}
        //         )
        //     )
        // );

        // Initialize random lattice
        std::vector<std::vector<std::vector<Vec3D>>> lattice = HM_local.initialize_lattice(L);

        // Set batchsize
        int batchsize = 100;

        // Thermalization
        for (int n = 0; n < Ntherm; ++n) {

            // Decide which step function to use
            for (int m = 0; m < Nsubsweep; ++m) HM_local.step(lattice, temp);
            // for (int m = 0; m < Nsubsweep; ++m) HM_local.step_ml(lattice, temp, model);
            // sweep_batched(HM_local, lattice, temp, model, Nsubsweep, batchsize);
        }
        
        // Start Simulation
        for (int n = 0; n < Nsample; ++n) {
            
            // Decide which step function to use
            for (int m = 0; m < Nsubsweep; ++m) HM_local.step(lattice, temp); // Regular step function
            //for (int m = 0; m < Nsubsweep; ++m) HM_local.step_ml(lattice, temp, model); // ML step function with sigle spin flip prediction
            //sweep_batched(HM_local, lattice, temp, model, Nsubsweep, batchsize); // ML step funciton with batch of spins flip prediction
            
            // Calculate Energy and Magentization
            double E = HM.total_energy(lattice) / (L * L * L);
            std::vector<double> Mvec = HM_local.total_magnetization(lattice) / (L * L * L);
            double Mmag = std::sqrt(dot(Mvec, Mvec));
    
            Energies.push_back(E);
            Energies2.push_back(E * E);
            Mags.push_back(Mmag);
            Mags2.push_back(Mmag * Mmag);
        }
    
        // Calculate means and squared means
        double E_mean = HM_local.energy_average(Energies);
        double E2_mean = HM_local.energy_average(Energies2);
        double M_mean = HM_local.energy_average(Mags);
        double M2_mean = HM_local.energy_average(Mags2);
        
        // Get specific heat and susceptibility
        double Cv = (E2_mean - E_mean * E_mean) / (temp * temp);
        double Chi = (M2_mean - M_mean * M_mean) / temp;
    
        // Save values
        E_arr[idx] = E_mean;
        M_arr[idx] = M_mean;
        chi_arr[idx] = Chi;
        cv_arr[idx] = Cv;
        
    }
    
    // Measure elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Duration of Parallelized code: " << elapsed.count() << " seconds." << std::endl;

    // Write all to CSV
    std::filesystem::create_directories("output");
    std::ofstream fout("output/heisenberg_ml_batch.csv");
    fout << "T,E_avg,M_avg_mag,chi,cv\n";
    for (size_t n = 0; n < Temperatures.size(); ++n) {
        fout 
            << Temperatures[n] << ","
            << E_arr[n] << ","
            << M_arr[n] << ","
            << chi_arr[n] << ","
            << cv_arr[n] << "\n";
    }
    fout.close();

    return 0;
}

