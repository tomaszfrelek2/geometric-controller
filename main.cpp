// main.cpp
#include "tricopter.h"
#include <cmath>

// Define our specific target behavior for this test
Target target_recovery(double t) {
    Target tgt;
    tgt.x << 0.0, 0.0, -2.0;
    tgt.v.setZero();
    tgt.a.setZero();
    tgt.b1 << 1.0, 0.0, 0.0;
    return tgt;
}

int main() {
    // Setup initial conditions (Upside Down)
    double angle = 178.0 * M_PI / 180.0;
    double c = std::cos(angle);
    double s = std::sin(angle);
    
    Eigen::Matrix3d R_178;
    R_178 << 1,  0,  0,
             0,  c, -s,
             0,  s,  c;

    State init_recovery;
    init_recovery.x << 0.0, 0.0, -5.0;
    init_recovery.R = R_178;

    // Run
    run_scenario("Clean Multi-File Recovery", init_recovery, target_recovery, 4.0);

    return 0;
}
