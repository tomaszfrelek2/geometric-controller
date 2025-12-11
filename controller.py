# ==========================================
# Geometric Tricopter Controller Simulation
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. MATH HELPER FUNCTIONS
# ==========================================

def hat(v):
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def expm_SO3(omega_vector):
    angle = np.linalg.norm(omega_vector)
    if angle < 1e-6:
        return np.eye(3)
    axis = omega_vector / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# ==========================================
# 2. SYSTEM PARAMETERS
# ==========================================

class TricopterParams:
    def __init__(self):
        self.m = 1.0          # Mass (kg)
        self.g = 9.81         # Gravity (m/s^2)
        self.dx = 0.4   
        self.dy = 0.4   
        self.dt = 0.5   

        # Inertia (Point mass approximation)
        m_arm = 0.2 
        jx = 2 * m_arm * (self.dy**2) 
        jy = 2 * m_arm * (self.dx**2) + m_arm * (self.dt**2)
        jz = jx + jy
        self.J = np.diag([jx, jy, jz]) 
        self.invJ = np.linalg.inv(self.J)
        self.e3 = np.array([0, 0, 1])

# ==========================================
# 3. CONTROL ALLOCATION 
# ==========================================

def allocation(f_desired, M_desired, params):
    """
    Robust allocation that prioritizes Torque when Thrust is negative.
    """
    
    target_f = max(0.0, f_desired)

    A = np.array([
        [1,          1,          1,           0],
        [-params.dy, params.dy,  0,           0],
        [params.dx,  params.dx, -params.dt,   0],
        [0,          0,          0,          -params.dt]
    ])

    target_vec = np.array([target_f, M_desired[0], M_desired[1], M_desired[2]])
    u_virt = np.linalg.pinv(A) @ target_vec

    f1 = u_virt[0]
    f2 = u_virt[1]
    f3_vert = u_virt[2]
    f3_lat = u_virt[3]

    
    f1 = max(0.0, f1)
    f2 = max(0.0, f2)
    
    # Add epsilon to vertical component to prevent divide-by-zero or rapid flipping 
    # when tail thrust is near zero.
    epsilon = 1e-4
    
   
    safe_vert = max(0.0, f3_vert) 
    
    raw_delta = np.arctan2(f3_lat, safe_vert + epsilon)
    
    limit = np.pi / 2.0 
    delta = np.clip(raw_delta, -limit, limit)
    
    # Magnitude
    f3 = np.sqrt(f3_vert**2 + f3_lat**2)
    f3 = max(0.0, f3)

    return f1, f2, f3, delta

# ==========================================
# 4. GEOMETRIC CONTROLLER
# ==========================================

def geometric_controller(state, target, params):
    x, v, R, Omega = state['x'], state['v'], state['R'], state['Omega']
    xd, vd, ad, b1d = target['x'], target['v'], target['a'], target['b1']

    # Gains
    wn_pos, zeta_pos = 2.0, 1.0
    kx = params.m * wn_pos**2             
    kv = params.m * 2 * zeta_pos * wn_pos 

    wn_rot, zeta_rot = 10.0, 0.9
    max_J = np.max(np.diag(params.J)) 
    kR_scalar = max_J * wn_rot**2              
    kOmega_scalar = max_J * 2 * zeta_rot * wn_rot 
    K_R = np.eye(3) * kR_scalar
    K_Omega = np.eye(3) * kOmega_scalar

    # Translation
    ex = x - xd
    ev = v - vd
    F_des = -kx * ex - kv * ev - params.m * params.g * params.e3 + params.m * ad

    # Attitude Construction
    norm_F_des = np.linalg.norm(F_des)
    b3d = np.array([0,0,1]) if norm_F_des < 1e-6 else -F_des / norm_F_des 

    cross_b3_b1 = np.cross(b3d, b1d)
    norm_cross = np.linalg.norm(cross_b3_b1)
    b2d = np.array([0,1,0]) if norm_cross < 1e-6 else cross_b3_b1 / norm_cross
    b1d_new = np.cross(b2d, b3d)
    Rd = np.column_stack((b1d_new, b2d, b3d))

    # Thrust
    current_b3 = R @ params.e3
    f = -np.dot(F_des, current_b3)

    # Rotation
    eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
    eOmega = Omega 
    gyroscopic = np.cross(Omega, params.J @ Omega)
    M = -K_R @ eR - K_Omega @ eOmega + gyroscopic

    return f, M

# ==========================================
# 5. DYNAMICS
# ==========================================

def dynamics(state, actuators, params):
    R, Omega = state['R'], state['Omega']
    f1, f2, f3, delta = actuators

    F_b = np.array([0.0, f3 * np.sin(delta), -(f1 + f2 + f3 * np.cos(delta))])
    M_b = np.array([
        params.dy * (f2 - f1),
        params.dx * (f1 + f2) - params.dt * f3 * np.cos(delta),
        -params.dt * f3 * np.sin(delta)
    ])

    v_dot = params.g * params.e3 + (R @ F_b) / params.m
    gyroscopic = np.cross(Omega, params.J @ Omega)
    Omega_dot = params.invJ @ (M_b - gyroscopic)
    
    return state['v'], v_dot, Omega_dot

# ==========================================
# 6. SCENARIO RUNNER
# ==========================================

def run_scenario(name, initial_state, target_fn, t_end=5.0):
    params = TricopterParams()
    dt = 0.01
    time = np.arange(0, t_end, dt)
    
    state = initial_state.copy()
    log = {'x': [], 'y': [], 'z': [], 'delta': [], 'time': []}
    snapshots = []

    print(f"--- Starting Scenario: {name} ---")

    for i, t in enumerate(time):
        target = target_fn(t)
        
        f_virt, M_virt = geometric_controller(state, target, params)
        f1, f2, f3, delta = allocation(f_virt, M_virt, params)
        dx, dv, dW = dynamics(state, [f1, f2, f3, delta], params)

        state['x'] += dx * dt
        state['v'] += dv * dt
        state['Omega'] += dW * dt
        state['R'] = state['R'] @ expm_SO3(state['Omega'] * dt)

        log['x'].append(state['x'][0])
        log['y'].append(state['x'][1])
        log['z'].append(state['x'][2])
        log['delta'].append(delta)
        log['time'].append(t)

        if i % 30 == 0: # Snapshot every 0.3s
            snapshots.append((state['x'].copy(), state['R'][:, 0]))

    return log, snapshots

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # --- Scenario 1: Normal Diagonal Flight ---
    init_normal = {
        'x': np.array([0.0, 0.0, 0.0]),
        'v': np.array([0.0, 0.0, 0.0]),
        'R': np.eye(3),
        'Omega': np.array([0.0, 0.0, 0.0])
    }
    def target_normal(t):
        return {
            'x': np.array([1.0, 1.0, -1.0]), 
            'v': np.zeros(3), 'a': np.zeros(3), 
            'b1': np.array([1.0, 1.0, 0.0])/np.sqrt(2)
        }
    
    # --- Scenario 2: Upside Down Recovery (178 degrees) ---
    # Construct 178 degree rotation about X-axis
    angle = np.radians(178)
    c, s = np.cos(angle), np.sin(angle)
    R_178 = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])
    
    init_recovery = {
        'x': np.array([0.0, 0.0, -5.0]), 
        'v': np.array([0.0, 0.0, 0.0]),
        'R': R_178,                      # 178 degree initial roll
        'Omega': np.array([0.0, 0.0, 0.0])
    }
    
    # Target: Stabilize at 2m high 
    def target_recovery(t):
        return {
            'x': np.array([0.0, 0.0, -2.0]), 
            'v': np.zeros(3), 'a': np.zeros(3), 
            'b1': np.array([1.0, 0.0, 0.0])
        }

    # Run Simulations
    log1, snap1 = run_scenario("Normal Flight", init_normal, target_normal, t_end=4.0)
    log2, snap2 = run_scenario("Upside Down Recovery", init_recovery, target_recovery, t_end=4.0)

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 8))

    # Plot 1: Normal Flight
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(log1['x'], log1['y'], log1['z'], label='Path', color='blue')
    ax1.scatter([1], [1], [-1], color='r', marker='x', s=50)
    for pos, head in snap1:
        ax1.quiver(pos[0], pos[1], pos[2], head[0], head[1], head[2], length=0.2, color='orange')
    ax1.set_title("Normal: Diagonal Path")
    ax1.invert_zaxis()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(log1['time'], np.degrees(log1['delta']), color='purple')
    ax2.set_title("Recovery: Servo Angle")
    ax2.set_ylabel("Deg"); ax2.set_xlabel("Time (s)")
    ax2.grid(True)
    ax2.set_ylim(-100, 100)

    # Plot 3: Altitude Comparison
    ax3 = fig.add_subplot(2, 2, 3)
    #ax3.plot(log1['time'], log1['z'], label='Normal')
    ax3.plot(log2['time'], log2['z'], label='Recovery Path')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Altitude (m)')
    ax3.set_title("Altitude")

    ax3.invert_yaxis()
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Recovery Servo Activity
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(log2['time'], np.degrees(log2['delta']), color='purple')
    ax4.set_title("Recovery: Servo Angle")
    ax4.set_ylabel("Deg"); ax4.set_xlabel("Time (s)")
    ax4.grid(True)
    ax4.set_ylim(-100, 100)

   
    plt.tight_layout()
    plt.show()
