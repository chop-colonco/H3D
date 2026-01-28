#include "Metropolis.h"
#include "helpers.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <cstdio>


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> lattice_size_dist(1, 20);
std::uniform_real_distribution<> temp_dist(0.1, 5.0);

std::uniform_real_distribution<> temp_core(1.2, 1.6);
std::uniform_real_distribution<> temp_wide(0.1, 5.0);
std::uniform_real_distribution<> temp_low(0.1, 0.5);
std::uniform_real_distribution<> prob_dist(0.0, 1.0);


int main() {

    if (std::remove("training_data_v2.csv") == 0) {
        std::cout << "Old training_data.csv removed.\n";
    }

    std::cout << "Start collecting data..." << std::endl;

    std::string output_path = "training_data.csv";
    std::ofstream fout(output_path);
    fout << "old_x,old_y,old_z,"
     << "new_x,new_y,new_z,"
     << "spin1_x,spin1_y,spin1_z,"
     << "spin2_x,spin2_y,spin2_z,"
     << "spin3_x,spin3_y,spin3_z,"
     << "spin4_x,spin4_y,spin4_z,"
     << "spin5_x,spin5_y,spin5_z,"
     << "spin6_x,spin6_y,spin6_z,"
     << "norm_latticesize,norm_temp,label\n";

    for (int n = 0; n < 100000; ++n) {

        if (n % 1000 == 0) std::cout << "At sample " << n << std::endl;

        int L = lattice_size_dist(gen);
        Heisenberg_Metropolis HM(L, gen);

        double temp;
        std::uniform_real_distribution<> temp_low(0.1, 0.5);

        if (prob_dist(gen) < 0.4) {
            temp = temp_core(gen);
        } else if (prob_dist(gen) < 0.8) {
            temp = temp_low(gen);
        } else {
            temp = temp_wide(gen);
        }


        double L_min = 2.0, L_max = 20.0;
        double T_min = 0.1, T_max = 5.0;

        double norm_L = (double(L) - L_min) / (L_max - L_min);
        double norm_temp = (temp - T_min) / (T_max - T_min);

        int Ntherm = HM.Ntherm;
        int Nsample = HM.Nsample;
        int Nsubsweep = HM.Nsubsweep;
    
        std::vector<std::vector<std::vector<Vec3D>>> lattice(L, std::vector<std::vector<Vec3D>>(L, std::vector<Vec3D>(L)));
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                for (int k = 0; k < L; ++k)
                    lattice[i][j][k] = HM.random_unit_vector();

        // Thermalize
        for (int l = 0; l < Ntherm; ++l) {
            HM.step(lattice, temp);
        }

        int i = HM.dist(HM.gen);
        int j = HM.dist(HM.gen);
        int k = HM.dist(HM.gen);
    
        Vec3D old_spin = lattice[i][j][k];
        Vec3D new_spin = HM.random_unit_vector();

        Vec3D spin1 = lattice[(i + L + 1) % L][j][k];
        Vec3D spin2 = lattice[(i + L - 1) % L][j][k];
        Vec3D spin3 = lattice[i][(j + L + 1) % L][k];
        Vec3D spin4 = lattice[i][(j + L - 1) % L][k];
        Vec3D spin5 = lattice[i][j][(k + L + 1) % L];
        Vec3D spin6 = lattice[i][j][(k + L - 1) % L];
    
        double old_energy = HM.local_energy(lattice, i, j, k);
        lattice[i][j][k] = new_spin;
        double new_energy = HM.local_energy(lattice, i, j, k);
        lattice[i][j][k] = old_spin;
    
        double delta_E = new_energy - old_energy;
    
        double R = HM.dist_accept(HM.gen);
        int accept;
        if (delta_E < 0 || R < std::exp(-delta_E / temp)) {
            lattice[i][j][k] = new_spin;
            accept = 1;
        } else {
            accept = 0;
        }

        double accept_prob = (delta_E < 0) ? 1.0 : std::exp(-delta_E / temp);

        fout << old_spin[0] << "," << old_spin[1] << "," << old_spin[2] << ","
        << new_spin[0] << "," << new_spin[1] << "," << new_spin[2] << ","
        << spin1[0] << "," << spin1[1] << "," << spin1[2] << ","
        << spin2[0] << "," << spin2[1] << "," << spin2[2] << ","
        << spin3[0] << "," << spin3[1] << "," << spin3[2] << ","
        << spin4[0] << "," << spin4[1] << "," << spin4[2] << ","
        << spin5[0] << "," << spin5[1] << "," << spin5[2] << ","
        << spin6[0] << "," << spin6[1] << "," << spin6[2] << ","
        << norm_L << "," << norm_temp << "," << accept_prob << "\n";
    }

    fout.close();
    std::cout << "\nFinished collecting samples into " << output_path << std::endl;

    return 0;
}