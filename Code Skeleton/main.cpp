#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
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
        double mA, double mB, // masses of body A and body b [in]
        double* Fx, double* Fy, double* Fz // force [out]
        ) {
            // TODO
}

#if PLOT
    void createScatter(double v[][3], double size){
        std::vector<double> xVec, yVec, zVec;
        for(int i = 0; i<size; i++){
            xVec.push_back(v[i][0]);
            yVec.push_back(v[i][1]);
            zVec.push_back(v[i][2]);
        }
        plt::scatter(xVec, yVec, zVec);
    }

    void displayGraphs(){
        plt::show();
    }
#endif

int main() {
    srand((unsigned) time(nullptr));
    double t = 0.0; // initial time
    double dt = 0.1; // time-step size
    double T = 2000.0; // final time
    int N = 900; // number of bodies
    auto force = new double[N][3]; // each of the forces

    // random number generator seeded with current clock time
    std::default_random_engine engine(
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    auto mass = new double[N]; // each of the body masses
    double mMean = 1.0, mStdDev = 0.0; // mean and standard deviation of mass values
    std::normal_distribution<double> massDistribution(mMean,mStdDev);
    for(int n = 0; n < N; n++) {
        mass[n] = massDistribution(engine);
    }

    auto p = new double[N][3]; // each of the body positions
    double pMean = 0.0, pStdDev = 1.0; // mean and standard deviation of position values
    std::normal_distribution<double> positionDistribution(pMean,pStdDev);
    for(int n = 0; n < N; n++) {
        p[n][0] = positionDistribution(engine);
        p[n][1] = positionDistribution(engine);
        p[n][2] = positionDistribution(engine);
    }

    #if PLOT
        createScatter(p, N); // plot initial positions of bodies
    #endif

    auto v = new double[N][3]; // each of the body velocities
    double vMean = 0.0, vStdDev = 1.0; // mean and standard deviation of velocity values
    std::normal_distribution<double> velocityDistribution(vMean,vStdDev);
    for (int n = 0; n < N; n++) {
        v[n][0] = velocityDistribution(engine);
        v[n][1] = velocityDistribution(engine);
        v[n][2] = velocityDistribution(engine);
    }

    // mark start time
    auto start = std::chrono::high_resolution_clock::now();

    while (t < T) { // we assume T is defined
        // TODO
        t += dt;
    }

    // measure runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time taken: %f seconds", duration.count()/1000000.0);

    #if PLOT
        createScatter(p, N);  // plot final positions of bodies
        displayGraphs(); // show all plots
    #endif

    delete[](mass);
    delete[](force);
    delete[](p);
    delete[](v);

}




