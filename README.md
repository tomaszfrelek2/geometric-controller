# Geometric Tricopter Controller on SE(3)


A geometric tracking control strategy for a standard Y-configuration tricopter UAV. This project implements a nonlinear controller defined globally on the Special Euclidean group $SE(3)$, capable of executing complex large-angle maneuvers and recovering from inverted flight conditions.

## 📝 Abstract

Unlike quadrotors, standard tricopters utilize a tilting rear rotor for yaw control. This introduces a nonlinear mechanical coupling where every yaw command generates a parasitic lateral force. 

This repository implements a **Geometric Control** framework that:
1.  **Decouples Dynamics:** Uses a nonlinear control allocation scheme to resolve servo angle $\delta$ and rear thrust $f_3$, isolating parasitic forces as bounded disturbances.
2.  **Avoids Singularities:** Defined on the configuration manifold $SE(3)$ (using rotation matrices) to avoid Gimbal lock and quaternion unwinding.
3.  **Guarantees Stability:**
    * **Attitude:** Exponential Stability.
    * **Position:** Uniformly Ultimately Bounded (UUB) stability in the presence of parasitic lateral forces.
4.  **Global Recovery:** Features a "geometric dimmer switch" on thrust that allows the drone to recover from inverted (upside-down) initial conditions.

## 🚀 Key Features

* **Nonlinear Control Allocation:** Analytical solution for the Y-configuration tricopter mixing matrix.
* **Large Angle Stabilization:** Recovery from $178^\circ$ roll (near-inverted) initial conditions.
* **Inverted Flight Logic:** Priority logic for control allocation when negative thrust is commanded during free-fall recovery.
* **Python Simulation:** Full 6-DOF rigid body physics simulation with actuator dynamics.
