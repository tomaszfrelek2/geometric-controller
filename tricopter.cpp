// tricopter.cpp
#include "tricopter.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace Eigen; 

// --- Struct Constructors ---

TricopterParams::TricopterParams() {
    m = 1.0;
    g = 9.81;
    dx = 0.4; dy = 0.4; dt = 0.5;

    double m_arm = 0.2;
    double jx = 2 * m_arm * (dy * dy);
    double jy = 2 * m_arm * (dx * dx) + m_arm * (dt * dt);
    double jz = jx + jy;
    
    J.setZero();
    J.diagonal() << jx, jy, jz;
    invJ = J.inverse();
    e3 << 0, 0, 1;
}

State::State() {
    x.setZero();
    v.setZero();
    R.setIdentity();
    Omega.setZero();
}

// --- Math Helpers ---

Matrix3d hat(const Vector3d& v) {
    Matrix3d S;
    S <<  0,    -v(2),  v(1),
          v(2),  0,    -v(0),
         -v(1),  v(0),  0;
    return S;
}

Vector3d vee(const Matrix3d& S) {
    return Vector3d(S(2, 1), S(0, 2), S(1, 0));
}

Matrix3d expm_SO3(const Vector3d& omega_vector) {
    double angle = omega_vector.norm();
    if (angle < 1e-6) return Matrix3d::Identity();
    Vector3d axis = omega_vector / angle;
    Matrix3d K = hat(axis);
    return Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * (K * K);
}

// --- Controller & Dynamics ---

Actuators allocation(double f_desired, const Vector3d& M_desired, const TricopterParams& params) {
    Matrix4d A;
    A << 1,          1,          1,          0,
        -params.dy,  params.dy,  0,          0,
         params.dx,  params.dx, -params.dt,  0,
         0,          0,          0,         -params.dt;

    Vector4d target_vec;
    target_vec << f_desired, M_desired(0), M_desired(1), M_desired(2);

    Vector4d u_virt = A.colPivHouseholderQr().solve(target_vec);
    
    double f3_vert = u_virt(2);
    double f3_lat  = u_virt(3);
    double f3 = std::sqrt(f3_vert * f3_vert + f3_lat * f3_lat);
    double delta = std::atan2(f3_lat, f3_vert);

    return {u_virt(0), u_virt(1), f3, delta};
}

std::pair<double, Vector3d> geometric_controller(const State& state, const Target& target, const TricopterParams& params) {
    double wn_pos = 2.0, zeta_pos = 1.0;
    double kx = params.m * wn_pos * wn_pos;
    double kv = params.m * 2.0 * zeta_pos * wn_pos;

    double wn_rot = 10.0, zeta_rot = 0.9;
    double max_J = params.J.diagonal().maxCoeff();
    double kR_scalar = max_J * wn_rot * wn_rot;
    double kOmega_scalar = max_J * 2.0 * zeta_rot * wn_rot;

    Vector3d ex = state.x - target.x;
    Vector3d ev = state.v - target.v;
    Vector3d F_des = -kx * ex - kv * ev - params.m * params.g * params.e3 + params.m * target.a;

    Vector3d b3d;
    if (F_des.norm() < 1e-6) b3d << 0, 0, 1;
    else b3d = -F_des.normalized();

    Vector3d b2d;
    Vector3d cross_b3_b1 = b3d.cross(target.b1);
    if (cross_b3_b1.norm() < 1e-6) b2d << 0, 1, 0;
    else b2d = cross_b3_b1.normalized();

    Vector3d b1d_new = b2d.cross(b3d);
    Matrix3d Rd;
    Rd << b1d_new, b2d, b3d;

    double f = -F_des.dot(state.R * params.e3);
    Vector3d eR = 0.5 * vee(Rd.transpose() * state.R - state.R.transpose() * Rd);
    Vector3d M = -(Matrix3d::Identity() * kR_scalar) * eR - (Matrix3d::Identity() * kOmega_scalar) * state.Omega + state.Omega.cross(params.J * state.Omega);

    return {f, M};
}

Derivatives dynamics(const State& state, const Actuators& acts, const TricopterParams& params) {
    Vector3d F_b;
    F_b << 0.0, acts.f3 * std::sin(acts.delta), -(acts.f1 + acts.f2 + acts.f3 * std::cos(acts.delta));

    Vector3d M_b;
    M_b << params.dy * (acts.f2 - acts.f1),
           params.dx * (acts.f1 + acts.f2) - params.dt * acts.f3 * std::cos(acts.delta),
           -params.dt * acts.f3 * std::sin(acts.delta);

    Vector3d v_dot = params.g * params.e3 + (state.R * F_b) / params.m;
    Vector3d Omega_dot = params.invJ * (M_b - state.Omega.cross(params.J * state.Omega));

    return {state.v, v_dot, Omega_dot};
}

void run_scenario(std::string name, State initial_state, TargetFunc target_fn, double t_end) {
    TricopterParams params;
    double dt = 0.01;
    State state = initial_state;
    
    std::ofstream log_file("simulation_log.csv");
    log_file << "time,x,y,z,delta_deg\n";
    std::cout << "--- Starting Scenario: " << name << " ---" << std::endl;

    for (double t = 0; t < t_end; t += dt) {
        Target target = target_fn(t);
        auto [f_virt, M_virt] = geometric_controller(state, target, params);
        Actuators acts = allocation(f_virt, M_virt, params);
        Derivatives deriv = dynamics(state, acts, params);

        state.x += deriv.dx * dt;
        state.v += deriv.dv * dt;
        state.Omega += deriv.dOmega * dt;
        state.R = state.R * expm_SO3(state.Omega * dt);

        log_file << t << "," << state.x(0) << "," << state.x(1) << "," << state.x(2) << "," << (acts.delta * 180.0 / M_PI) << "\n";
    }
    log_file.close();
    std::cout << "Done." << std::endl;
}
