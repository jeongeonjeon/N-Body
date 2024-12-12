#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <../lib/x86_64-linux-gnu/openmpi/include/mpi.h>

#define PLOT true
#if PLOT
    #include "matplotlibcpp.h"
    namespace plt = matplotlibcpp;
#endif

// #define G 6.6742e-11 // Gravitational constant in SI units
#define G 1.0 // Gravitational constant

inline void calcForce(
        double pAx, double pAy, double pAz, // position of body A [in]
        double pBx, double pBy, double pBz, // position of body B [in]
        double mA, double mB,               // masses of body A and body b [in]
        double* Fx, double* Fy, double* Fz // force [out]
        ) {

    // TODO
    double dx = pBx - pAx;
    double dy = pBy - pAy;
    double dz = pBz - pAz;
    double r2 = dx * dx + dy * dy + dz * dz;
    double r, F;

    #if r2 != 0
        r = sqrt(r2);
        F = G * mA * mB / r2;

        *Fx = F * (dx / r);
        *Fy = F * (dy / r);
        *Fz = F * (dz / r);
    #endif
}

#if PLOT
    void createScatter(double v[][3], double size) {
        std::vector<double> xVec, yVec, zVec;
        for (int i = 0; i < size; i++) {
            xVec.push_back(v[i][0]);
            yVec.push_back(v[i][1]);
            zVec.push_back(v[i][2]);
        }
        plt::scatter(xVec, yVec, zVec);
    }

    void displayGraphs() {
        plt::show();
    }
#endif

int main() {
    // srand((unsigned) time(nullptr));
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t = 0.0; // initial time
    double dt = 0.1; // time-step size
    double T = 1000.0; // final time
    int N = 1000; // number of bodies
    auto force = new double[N][3]; // each of the forces

    // random number generator seeded with current clock time
    std::default_random_engine engine(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    auto mass = new double[N]; // each of the body masses
    auto p = new double[N][3]; // each of the body positions
    auto v = new double[N][3]; // each of the body velocities

    double mMean = 1.0, mStdDev = 0.0; // mean and standard deviation of mass values
    std::normal_distribution<double> massDistribution(mMean,mStdDev);

    double pMean = 0.0, pStdDev = 1.0; // mean and standard deviation of position values
    std::normal_distribution<double> positionDistribution(pMean, pStdDev);

    double vMean = 0.0, vStdDev = 1.0; // mean and standard deviation of velocity values
    std::normal_distribution<double> velocityDistribution(vMean, vStdDev);


    for (int n = 0; n < N; n++) {
        mass[n] = massDistribution(engine);
        p[n][0] = positionDistribution(engine);
        p[n][1] = positionDistribution(engine);
        p[n][2] = positionDistribution(engine);
        v[n][0] = velocityDistribution(engine);
        v[n][1] = velocityDistribution(engine);
        v[n][2] = velocityDistribution(engine);
    }

    #if PLOT
        if (rank == 0) {
            createScatter(p, N); // plot initial positions of bodies
        }
    #endif

    // Divide work among MPI ranks
    int local_N = N / size;
    int start_idx = rank * local_N;
    int end_idx = (rank == size - 1) ? N : start_idx + local_N;

    // mark start time
    auto start = std::chrono::high_resolution_clock::now();

    while (t < T) { // we assume T is defined
        // TODO
        // Reset forces
        #pragma omp parallel for simd
        for (int i = start_idx; i < end_idx; i++) {
            force[i][0] = force[i][1] = force[i][2] = 0.0;
        }

        // Calculate forces
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i = start_idx; i < end_idx; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    double Fx, Fy, Fz;
                    calcForce(p[i][0], p[i][1], p[i][2],
                            p[j][0], p[j][1], p[j][2],
                            mass[i], mass[j],
                            &Fx, &Fy, &Fz);
                    force[i][0] += Fx;
                    force[i][1] += Fy;
                    force[i][2] += Fz;
                }
            }
        }

        // Gather forces across ranks
        MPI_Allreduce(MPI_IN_PLACE, force, 3 * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update positions and velocities
        #pragma omp parallel for
        for (int i = start_idx; i < end_idx; i++) {
            v[i][0] += dt * force[i][0] / mass[i];
            v[i][1] += dt * force[i][1] / mass[i];
            v[i][2] += dt * force[i][2] / mass[i];

            p[i][0] += dt * v[i][0];
            p[i][1] += dt * v[i][1];
            p[i][2] += dt * v[i][2];
        }

        t += dt;
    }

    // measure runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (rank == 0) {
        printf("Time taken: %f seconds\n", duration.count() / 1000000.0);
    }

    #if PLOT
        if (rank == 0) {
            std::cout << "Creating scatter plot..." << std::endl;
            createScatter(p, N);  // plot final positions of bodies
            std::cout << "Displaying plot..." << std::endl;
            displayGraphs(); // show all plots
        }
    #endif

    delete[](mass);
    delete[](force);
    delete[](p);
    delete[](v);

    MPI_Finalize();

    return 0;
}