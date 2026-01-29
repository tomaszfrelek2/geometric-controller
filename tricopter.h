// tricopter.h
#ifndef TRICOPTER_H
#define TRICOPTER_H

#include <Eigen/Dense>
#include <vector>
#include <string>

// --- Data Structures ---

struct TricopterParams {
    double m;
    double g;
    double dx, dy, dt;
    Eigen::Matrix3d J;
    Eigen::Matrix3d invJ;
    Eigen::Vector3d e3;

    TricopterParams(); // Constructor declaration
};

struct State {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
    Eigen::Matrix3d R;
    Eigen::Vector3d Omega;

    State();
};

struct Target {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
    Eigen::Vector3d a;
    Eigen::Vector3d b1;
};

struct Actuators {
    double f1, f2, f3, delta;
};

struct Derivatives {
    Eigen::Vector3d dx;
    Eigen::Vector3d dv;
    Eigen::Vector3d dOmega;
};

// --- Function Declarations ---

// Math Helpers
Eigen::Matrix3d hat(const Eigen::Vector3d& v);
Eigen::Vector3d vee(const Eigen::Matrix3d& S);
Eigen::Matrix3d expm_SO3(const Eigen::Vector3d& omega_vector);

// Controller & Dynamics
Actuators allocation(double f_desired, const Eigen::Vector3d& M_desired, const TricopterParams& params);

std::pair<double, Eigen::Vector3d> geometric_controller(const State& state, 
                                                        const Target& target, 
                                                        const TricopterParams& params);

Derivatives dynamics(const State& state, 
                     const Actuators& acts, 
                     const TricopterParams& params);

// Simulation Runner
typedef Target (*TargetFunc)(double);
void run_scenario(std::string name, State initial_state, TargetFunc target_fn, double t_end);

#endif // TRICOPTER_H
