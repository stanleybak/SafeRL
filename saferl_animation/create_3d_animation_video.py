#!/usr/bin/env python3
"""
3D Animation of F-16 Wingman Scenario with Ghost Prediction
Shows neural network prediction vs actual commanded trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.io import loadmat
import sys
import time
import onnxruntime as ort
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import RK45
import os
import math
import pickle

# Global configuration variables
LIVE_PLOT = False  # Set to True to visualize live in matplotlib instead of saving video

FPS = 30  # Frames per second - match simulation timestep for live plot
CYLINDER_HEIGHT = 500  # Height of 100ft boundary cylinder
CYLINDER_RADIUS = 100  # 100ft radius
GHOST_ALPHA = 0.1  # Transparency for ghost aircraft
PAUSE_FRAMES = FPS  # Number of frames to pause when showing command (1 second at 30fps)

# Add path for imports
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SafeRL'))

# Import necessary modules with proper path handling
#import importlib.util

sys.path.append('SafeRL')

from saferl.aerospace.models.f16.aerobench.example.wingman.wingman_autopilot import WingmanAutopilot
from saferl.aerospace.models.f16.aerobench.util import StateIndex, get_state_names, Euler, Freezable, get_script_path
from saferl.aerospace.models.f16.aerobench.highlevel.controlled_f16 import controlled_f16
from saferl.aerospace.models.f16.aerobench.visualize.anim3d import scale3d, rotate3d

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    'get xd, u, Nz, ps, Ny_r at the current time / state'
    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]
        xd_val, u_val, Nz_val, ps_val, Ny_r_val = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)
        xd_tup.append(xd_val)
        u_tup.append(u_val)
        Nz_tup.append(Nz_val)
        ps_tup.append(ps_val)
        Ny_r_tup.append(Ny_r_val)

    if num_aircraft == 1:
        return xd_tup[0], u_tup[0], Nz_tup[0], ps_tup[0], Ny_r_tup[0]
    return tuple(xd_tup), tuple(u_tup), tuple(Nz_tup), tuple(ps_tup), tuple(Ny_r_tup)

# Helper functions from plot_rollout_2d_interactive.py
def vec2magnorm(vec):
    """Convert 2D vector to magnitude-normalized form [mag, x/norm, y/norm]"""
    norm = np.linalg.norm(vec)
    if norm < 1e-6: 
        return np.array([0, 0, 0])
    return np.concatenate(([norm], vec / norm))

def get_observation(lead_state_13dim, wing_state_13dim):
    """Convert state vectors to normalized observation using magnorm mode"""
    lead_pos_e = lead_state_13dim[StateIndex.POS_E]
    lead_pos_n = lead_state_13dim[StateIndex.POS_N]
    lead_psi_angle = lead_state_13dim[StateIndex.PSI]
    lead_vt_val = lead_state_13dim[StateIndex.VT]
    wing_pos_e = wing_state_13dim[StateIndex.POS_E]
    wing_pos_n = wing_state_13dim[StateIndex.POS_N]
    wing_psi_angle = wing_state_13dim[StateIndex.PSI]
    wing_vt_val = wing_state_13dim[StateIndex.VT]
    lead_heading_rad = np.pi/2 - lead_psi_angle
    wing_heading_rad = np.pi/2 - wing_psi_angle
    lead_vel_vec = np.array([lead_vt_val * np.cos(lead_heading_rad), lead_vt_val * np.sin(lead_heading_rad)])
    wing_vel_vec = np.array([wing_vt_val * np.cos(wing_heading_rad), wing_vt_val * np.sin(wing_heading_rad)])
    cos_w_h, sin_w_h = np.cos(wing_heading_rad), np.sin(wing_heading_rad)
    rot_matrix_g2w = np.array([[cos_w_h, sin_w_h], [-sin_w_h, cos_w_h]])
    rel_pos_lead_wrt_wing_global = np.array([lead_pos_e - wing_pos_e, lead_pos_n - wing_pos_n])
    rejoin_offset_dist = 500
    rejoin_angle_rad = lead_heading_rad - np.deg2rad(180 - 60)
    rejoin_pos_global = np.array([lead_pos_e + rejoin_offset_dist * np.cos(rejoin_angle_rad),
                                lead_pos_n + rejoin_offset_dist * np.sin(rejoin_angle_rad)])
    rel_pos_rejoin_wrt_wing_global = rejoin_pos_global - np.array([wing_pos_e, wing_pos_n])
    rel_pos_lead_wrt_wing_wingframe = rot_matrix_g2w @ rel_pos_lead_wrt_wing_global
    rel_pos_rejoin_wrt_wing_wingframe = rot_matrix_g2w @ rel_pos_rejoin_wrt_wing_global
    wing_vel_wingframe = rot_matrix_g2w @ wing_vel_vec
    lead_vel_wingframe = rot_matrix_g2w @ lead_vel_vec
    norm_rel_pos_lead = vec2magnorm(rel_pos_lead_wrt_wing_wingframe)
    norm_rel_pos_rejoin = vec2magnorm(rel_pos_rejoin_wrt_wing_wingframe)
    norm_wing_vel = vec2magnorm(wing_vel_wingframe)
    norm_lead_vel = vec2magnorm(lead_vel_wingframe)
    
    # Build observation using actual magnitudes from vec2magnorm
    obs = np.array([
        norm_rel_pos_lead[0], norm_rel_pos_lead[1], norm_rel_pos_lead[2],
        norm_rel_pos_rejoin[0], norm_rel_pos_rejoin[1], norm_rel_pos_rejoin[2],
        norm_wing_vel[0], norm_wing_vel[1], norm_wing_vel[2],
        norm_lead_vel[0], norm_lead_vel[1], norm_lead_vel[2]
    ])
    
    # Apply normalization as per DubinsObservationProcessor
    normalization = np.array([1000, 1, 1, 1000, 1, 1, 400, 1, 1, 400, 1, 1])
    obs = obs / normalization
    
    # Apply clipping as per DubinsObservationProcessor
    obs = np.clip(obs, -1, 1)
    
    return obs

def get_network_action(obs, output_scaling=True):
    """Get action from neural network"""
    obs_exp = np.expand_dims(obs, axis=0).astype(np.float32)
    action = ort_session.run(output_names, {input_name: obs_exp})[0]

    if not output_scaling:
        return action[0][0], action[0][2]

    heading_output = np.clip(action[0][0], -1, 1)
    velocity_output = np.clip(action[0][2], -1, 1)
    heading_change = heading_output * 0.174533
    velocity_change = velocity_output * 10.0
    return heading_change, velocity_change

class SimModelError(RuntimeError):
    '''error during simulation'''

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft, f"{full_state.size} // {num_vars} == {full_state.size // num_vars} != {num_aircraft}"

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]

            alpha = state[StateIndex.ALPHA]
            if not (-0.349066 < alpha < 0.785398): # Alpha limits from F16constants, approx -20 to 45 deg
                 raise SimModelError(f"alpha ({alpha}) out of bounds")

            vel = state[StateIndex.VT]
            if not (100 < vel < 2500): # Velocity limits from F16constants
                 raise SimModelError(f"velocity ({vel}) out of bounds")

            alt = state[StateIndex.ALT]
            if not (-1000 < alt < 100000): # Altitude limits
                 raise SimModelError(f"altitude ({alt}) out of bounds")

            u_ref = u_refs[4*i:4*(i+1)]

            xd_val = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd_val)

        rv = np.hstack(xds)

        return rv

    return der_func

class F16SimState(Freezable):
    'object containing simulation state'
    def __init__(self, initial_state, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True, keep_intermediate_states=True,
                 custom_stop_func=None, integrator_kwargs=None):

        self.model_str = model_str = ap.llc.model_str
        self.v2_integrators = v2_integrators
        initial_state = np.array(initial_state, dtype=float)

        self.keep_intermediate_states = keep_intermediate_states
        self.custom_stop_func = custom_stop_func

        self.step = step
        self.ap = ap
        self.print_errors = print_errors

        llc = ap.llc
        num_vars = len(get_state_names()) + llc.get_num_integrators()

        if initial_state.size < num_vars:
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"
        self.x0 = x0

        self.times = None
        self.states = None
        self.modes = None
        self.extended_states = extended_states

        if self.extended_states:
            self.xd_list = None; self.u_list = None; self.Nz_list = None; self.ps_list = None; self.Ny_r_list = None

        self.cur_sim_time = 0
        self.total_sim_time = 0
        self.der_func = make_der_func(self.ap, self.model_str, self.v2_integrators)

        if integrator_kwargs is not None:
            self.integrator_kwargs = integrator_kwargs
        elif integrator_str == 'rk45':
            integrator_class = RK45
            self.integrator_kwargs = {'rtol': 1e-5, 'atol': 1e-8} 
        else:
            assert integrator_str == 'euler'
            integrator_class = Euler
            self.integrator_kwargs = {'step': step}
        
        if integrator_str == 'rk45': self.integrator_class = RK45
        else: self.integrator_class = Euler
            
        self.integrator = None
        self.freeze_attrs()

    def init_simulation(self):
        'initial simulation (upon first call to simulate_to)'
        assert self.integrator is None
        self.times = [0.0]
        self.states = [self.x0.copy()]

        self.ap.advance_discrete_mode(self.times[-1], self.states[-1])
        self.modes = [self.ap.mode]

        if self.extended_states:
            xd_val, u_val, Nz_val, ps_val, Ny_r_val = get_extended_states(self.ap, self.times[-1], self.states[-1],
                                                      self.model_str, self.v2_integrators)
            self.xd_list = [xd_val]; self.u_list = [u_val]; self.Nz_list = [Nz_val]; self.ps_list = [ps_val]; self.Ny_r_list = [Ny_r_val]
        
        self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1].copy(), np.inf,
                                                **self.integrator_kwargs)

    def simulate_to(self, tmax, tol=1e-7, update_mode_at_start=False):
        'simulate up to the passed in time'
        oldsettings = np.geterr()
        np.seterr(all='raise', under='ignore')
        start_time = time.perf_counter()
        ap = self.ap
        step = self.step

        if self.integrator is None:
            self.init_simulation()
        elif update_mode_at_start:
            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes[-1] = ap.mode
            if mode_changed:
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1].copy(), np.inf,
                                                        **self.integrator_kwargs)
        
        while True:
            current_integrator_time = self.integrator.t
            if current_integrator_time >= tmax - tol:
                break

            if self.integrator.status != 'running':
                break
            
            self.integrator.step()

        if abs(self.integrator.t - tmax) < tol:
            final_state_at_tmax = self.integrator.y.copy()
        else:
            if hasattr(self.integrator, 'dense_output') and callable(self.integrator.dense_output):
                dense_output = self.integrator.dense_output()
                if dense_output is not None:
                    final_state_at_tmax = dense_output(tmax).copy()
                else:
                     final_state_at_tmax = self.integrator.y.copy()

        self.times.append(tmax)
        self.states.append(final_state_at_tmax)
        
        ap.advance_discrete_mode(self.times[-1], self.states[-1])
        self.modes.append(ap.mode)

        if self.extended_states:
            xd_val, u_val, Nz_val, ps_val, Ny_r_val = get_extended_states(ap, self.times[-1], self.states[-1], self.model_str, self.v2_integrators)
            self.xd_list.append(xd_val); self.u_list.append(u_val); self.Nz_list.append(Nz_val); self.ps_list.append(ps_val); self.Ny_r_list.append(Ny_r_val)
        
        stop_func = self.custom_stop_func if self.custom_stop_func is not None else ap.is_finished
        if stop_func(self.times[-1], self.states[-1]):
            self.integrator.status = 'autopilot finished'
            
        if self.integrator.status == 'failed' and self.print_errors:
            print(f'Warning: integrator status was "{self.integrator.status}"')
        
        self.total_sim_time += time.perf_counter() - start_time
        np.seterr(**oldsettings)

# Load data and neural network
print("Loading data...")
m = loadmat('RTA_825_100.mat')
xd = m['xd'].T
udm = m['udm']
obsd = m['obsd']
td = m['td']
actionsd = m['actionsd']

# Load neural network
print("Loading neural network...")
ort_session = ort.InferenceSession('ckpt_825.onnx')
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]

# Load F-16 3D model data
print("Loading F-16 3D model...")
parent = get_script_path(__file__)
plane_point_data = os.path.join('SafeRL/saferl/aerospace/models/f16/aerobench/visualize', 'f-16.mat')
data = loadmat(plane_point_data)
f16_pts = data['V']
f16_faces = data['F']

def find_command_difference_sec(tolerance=1e-5):
    """Find the first step where NN command differs from UD command"""
    for step in range(udm.shape[0]): # was min(100, udm.shape[0])
        # Get states
        lead_state = xd[:13, step]
        wing_state = xd[16:29, step]
        
        # Get observation and NN action
        obs = get_observation(lead_state, wing_state)
        nn_heading, nn_velocity = get_network_action(obs)
        
        # Get actual commands
        ud_heading = udm[step, 0]
        ud_velocity = udm[step, 1]
        
        # Check if they differ
        if abs(nn_heading - ud_heading) > tolerance or abs(nn_velocity - ud_velocity) > tolerance:
            print(f"Found command difference at step {step}")
            print(f"  NN: heading={nn_heading:.6f}, velocity={nn_velocity:.6f}")
            print(f"  UD: heading={ud_heading:.6f}, velocity={ud_velocity:.6f}")
            return step
    
    raise ValueError("No command difference found")

def get_cache_filename(cache_type, initial_step=None, duration=None, use_nn=None):
    """Generate cache filename based on parameters"""
    if cache_type == "diff_step":
        return "cache_diff_step.pkl"
    elif cache_type == "trajectory":
        nn_str = "nn" if use_nn else "ud"
        return f"cache_traj_{initial_step}_{duration}_{nn_str}.pkl"
    return None

def load_from_cache(cache_type, **kwargs):
    """Load data from cache file if it exists"""
    filename = get_cache_filename(cache_type, **kwargs)
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {cache_type} from cache: {filename}")
                return data
        except Exception as e:
            print(f"Failed to load cache {filename}: {e}")
    return None

def save_to_cache(data, cache_type, **kwargs):
    """Save data to cache file"""
    filename = get_cache_filename(cache_type, **kwargs)
    if filename:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
                print(f"Saved {cache_type} to cache: {filename}")
        except Exception as e:
            print(f"Failed to save cache {filename}: {e}")

def simulate_trajectory(initial_step, duration, use_nn_commands=False, lead_init=None, wing_init=None):
    """Simulate aircraft trajectory from initial_step for duration seconds"""
    # Try to load from cache
    cached_result = load_from_cache("trajectory", initial_step=initial_step, 
                                   duration=duration, use_nn=use_nn_commands)
    if cached_result is not None:
        print(f"Loaded cached trajectory: step={initial_step}, duration={duration}, nn={use_nn_commands}")
        print(f"loaded {len(cached_result['lead_states'])} states")

        return cached_result
    
    print(f"Computing trajectory: step={initial_step}, duration={duration}, nn={use_nn_commands}")
    
    # Get initial states
    if initial_step == 0:
        assert lead_init is None and wing_init is None
        lead_initial_state = xd[:16, initial_step].copy()
        wing_initial_state = xd[16:32, initial_step].copy()
    else:
        assert lead_init is not None and wing_init is not None
        lead_initial_state = lead_init.copy()
        wing_initial_state = wing_init.copy()
    
    # Setup lead autopilot
    lead_ap_target_hdg_rad = xd[StateIndex.PSI, 0]
    lead_ap_target_vel_fps = xd[StateIndex.VT, 0]
    lead_ap_target_alt_ft = xd[StateIndex.ALT, 0]
    lead_autopilot = WingmanAutopilot(
        target_heading=lead_ap_target_hdg_rad,
        target_vel=lead_ap_target_vel_fps,
        target_alt=lead_ap_target_alt_ft,
        stdout=False
    )
    lead_f16_sim = F16SimState(
        initial_state=lead_initial_state.copy(),
        ap=lead_autopilot,
        step=1/60,  # 1/30 second timestep
        extended_states=False,
        integrator_str='rk45'
    )
    lead_f16_sim.init_simulation()
    
    # Setup wingman autopilot
    if initial_step > 0:
        wingman_ap_target_hdg_rad = xd[35, initial_step]
        wingman_ap_target_vel_fps = xd[36, initial_step]
        wingman_ap_target_alt_ft = xd[37, initial_step]
    else:
        wingman_ap_target_hdg_rad = xd[16 + StateIndex.PSI, 0]
        wingman_ap_target_vel_fps = xd[16 + StateIndex.VT, 0]
        wingman_ap_target_alt_ft = xd[16 + StateIndex.ALT, 0]
    
    wingman_autopilot = WingmanAutopilot(
        target_heading=wingman_ap_target_hdg_rad,
        target_vel=wingman_ap_target_vel_fps,
        target_alt=wingman_ap_target_alt_ft,
        stdout=False
    )
    wingman_f16_sim = F16SimState(
        initial_state=wing_initial_state.copy(),
        ap=wingman_autopilot,
        step=1/60,  # 1/60 second timestep
        extended_states=False,
        integrator_str='rk45'
    )
    wingman_f16_sim.init_simulation()
    
    # Storage for trajectory
    trajectory = {
        'lead_states': [lead_initial_state.copy()],
        'wing_states': [wing_initial_state.copy()],
        'times': [0.0],
        'commands': []
    }
    
    current_lead_sim_state = lead_initial_state.copy()
    current_wing_sim_state = wing_initial_state.copy()

    def update_autopilot_commands(hdg_chg, vel_chg):
        wingman_autopilot.targets[0] -= hdg_chg

        v_min_wingman = 700.0
        v_max_wingman = 900.0
        potential_new_target_vel = wingman_autopilot.targets[1] + vel_chg
        if potential_new_target_vel < v_min_wingman:
            wingman_autopilot.targets[1] = v_min_wingman
        elif potential_new_target_vel > v_max_wingman:
            wingman_autopilot.targets[1] = v_max_wingman
        else:
            wingman_autopilot.targets[1] = potential_new_target_vel

    # update command at time zero
    if use_nn_commands:
        current_obs = get_observation(current_lead_sim_state, current_wing_sim_state)
        nn_cmd_hdg_chg, nn_cmd_vel_chg = get_network_action(current_obs)

        update_autopilot_commands(nn_cmd_hdg_chg, nn_cmd_vel_chg)
        trajectory['commands'].append([nn_cmd_hdg_chg, nn_cmd_vel_chg])
    else:
        cmd_hdg_chg = udm[0, 0]
        cmd_vel_chg = udm[0, 1]
        update_autopilot_commands(cmd_hdg_chg, cmd_vel_chg)
        trajectory['commands'].append([cmd_hdg_chg, cmd_vel_chg])
    
    
    for sec in range(1, duration + 1):
        print(f"simulating up to second: {sec}")



        # Simulate both aircraft
        for micro_step in range(1, 61):
            target_sim_time = (sec - 1) + micro_step / 60
            print(f"{micro_step}: {target_sim_time}")
            update_mode_at_start = micro_step == 0

            lead_f16_sim.simulate_to(target_sim_time, update_mode_at_start=False)
            wingman_f16_sim.simulate_to(target_sim_time, update_mode_at_start=update_mode_at_start)
            
            # update states[-1] to be the state in xd to prevent drift
            #if not use_nn_commands and micro_step == 60:
            #    lead_f16_sim.states[-1] = xd[:16, sec].copy()
            #    wingman_f16_sim.states[-1] = xd[16:32, sec].copy()

            lead_state = lead_f16_sim.states[-1][:16].copy()
            wing_state = wingman_f16_sim.states[-1][:16].copy()

            trajectory['lead_states'].append(lead_state)
            trajectory['wing_states'].append(wing_state)
            trajectory['times'].append(target_sim_time)

        if sec == duration:
            break

        print(f"UPDATING COMMAND")
        
        if use_nn_commands:
            current_obs = get_observation(lead_state, wing_state)
            nn_cmd_hdg_chg, nn_cmd_vel_chg = get_network_action(current_obs)

            update_autopilot_commands(nn_cmd_hdg_chg, nn_cmd_vel_chg)
            trajectory['commands'].append([nn_cmd_hdg_chg, nn_cmd_vel_chg])
        else:
            cmd_hdg_chg = udm[sec, 0]
            cmd_vel_chg = udm[sec, 1]
            update_autopilot_commands(cmd_hdg_chg, cmd_vel_chg)
            trajectory['commands'].append([cmd_hdg_chg, cmd_vel_chg])

            # double check that the command is the same as the xd command
            current_obs = get_observation(lead_state, wing_state)
            nn_cmd_hdg_chg, nn_cmd_vel_chg = get_network_action(current_obs)

            #tol = 1e-2
            #if not np.allclose(nn_cmd_hdg_chg, cmd_hdg_chg, atol=tol) or not np.allclose(nn_cmd_vel_chg, cmd_vel_chg, atol=tol):
            #    print(f"NN command at time {sec} does not match the xd command")
            #    print(f"NN command: {nn_cmd_hdg_chg}, {nn_cmd_vel_chg}")
            #    print(f"XD command: {cmd_hdg_chg}, {cmd_vel_chg}")
            #    raise ValueError("NN command at time {sec} does not match the xd command")
            #else:
            #    print(f"NN command at time {sec} matches the xd command")
        
        
        # Check if within 100ft boundary (only for ghost)
        if use_nn_commands:
            lead_pos = np.array([lead_state[StateIndex.POS_E], 
                                lead_state[StateIndex.POS_N],
                                lead_state[StateIndex.ALT]])
            wing_pos = np.array([wing_state[StateIndex.POS_E],
                                wing_state[StateIndex.POS_N],
                                wing_state[StateIndex.ALT]])
            distance = np.linalg.norm(lead_pos - wing_pos)
            print(f"distance at time {sec:.2f}: {distance}")
            
            if distance < CYLINDER_RADIUS:
                print(f"Ghost aircraft entered 100ft boundary at time {sec:.2f}s, distance={distance:.1f}ft")
                break
    
    # Save to cache before returning
    
    save_to_cache(trajectory, "trajectory", initial_step=initial_step, 
                  duration=duration, use_nn=use_nn_commands)
    
    return trajectory

def draw_f16_aircraft(ax, state, color='black', alpha=1.0, scale=25):
    """Draw F-16 aircraft at given state"""
    # Extract position and orientation
    dx = state[StateIndex.POS_E]
    dy = state[StateIndex.POS_N]
    dz = state[StateIndex.ALT]
    phi = state[StateIndex.PHI]
    theta = state[StateIndex.THETA]
    psi = state[StateIndex.PSI]
    
    # Scale and rotate F-16 points
    pts = scale3d(f16_pts, [-scale, scale, scale])
    pts = rotate3d(pts, theta, psi - np.pi/2, -phi)
    
    # Create faces for the aircraft
    verts = []
    for count, face in enumerate(f16_faces):
        if LIVE_PLOT and count % 25 != 0: # draw every 25th face if live plot
            continue

        face_pts = []
        for findex in face:
            face_pts.append((pts[findex-1][0] + dx,
                           pts[findex-1][1] + dy,
                           pts[findex-1][2] + dz))
        verts.append(face_pts)
    
    # Create and add the aircraft mesh
    if color == 'red':
        face_color = (0.8, 0.2, 0.2, alpha)
        edge_color = (0.6, 0.1, 0.1, alpha)
    elif color == 'black':
        face_color = (0.2, 0.2, 0.2, alpha)
        edge_color = (0.1, 0.1, 0.1, alpha)
    else:  # default gray
        face_color = (0.5, 0.5, 0.5, alpha)
        edge_color = (0.3, 0.3, 0.3, alpha)
    
    aircraft_mesh = Poly3DCollection(verts, facecolors=face_color, 
                                   edgecolors=edge_color, alpha=alpha)
    ax.add_collection3d(aircraft_mesh)
    
    return aircraft_mesh

def draw_cylinder(ax, center, radius, height, color='red', alpha=0.3):
    """Draw a transparent cylinder"""
    # Create cylinder vertices
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(center[2] - height/2, center[2] + height/2, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    
    # Draw cylinder surface
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)
    
    # Draw top and bottom circles
    for z_val in [center[2] - height/2, center[2] + height/2]:
        x_circle = radius * np.cos(theta) + center[0]
        y_circle = radius * np.sin(theta) + center[1]
        z_circle = np.full_like(theta, z_val)
        ax.plot(x_circle, y_circle, z_circle, color=color, alpha=alpha*2)

def create_animation():
    """Create the 3D animation"""
    # Find step where commands differ
    diff_second = find_command_difference_sec()
    diff_step = diff_second * 2*FPS # convert to frame number
    print(f"diff_second: {diff_second}, Diff step: {diff_step}")
    
    # Simulate trajectories
    print("Simulating actual trajectory (0-24 steps)...")
    actual_traj = simulate_trajectory(0, 24, use_nn_commands=False)

    print(f"Actual trajectory length: {len(actual_traj['lead_states'])}")
    del actual_traj['commands']

    # before diff_second, take every 4th frame of actual_traj, and 
    # after concat with every frame of actual_traj
    split_index = diff_second*FPS*2
    print(f"split_index: {split_index}")

    SPLICE_FACTOR = 5
    lead_before_diff = actual_traj['lead_states'][:split_index:SPLICE_FACTOR]
    wing_before_diff = actual_traj['wing_states'][:split_index:SPLICE_FACTOR]
    times_before_diff = actual_traj['times'][:split_index:SPLICE_FACTOR]

    diff_step = len(lead_before_diff)
    print(f"updated diff_step to: {diff_step}")

    lead_after_diff = actual_traj['lead_states'][split_index:]
    wing_after_diff = actual_traj['wing_states'][split_index:]
    times_after_diff = actual_traj['times'][split_index:]

    print(f"lead_before_diff: {len(lead_before_diff)}, lead_after_diff: {len(lead_after_diff)}")

    actual_traj['lead_states'] = np.concatenate([lead_before_diff, lead_after_diff])
    actual_traj['wing_states'] = np.concatenate([wing_before_diff, wing_after_diff])
    actual_traj['times'] = np.concatenate([times_before_diff, times_after_diff])

    print(f"Actual trajectory length after concat: {len(actual_traj['lead_states'])}")

    # check that the simulated wing state at all times before diff_second matches
    #for sec in range(diff_second):
    #    micro_step = sec * 60
    #    wing_state = actual_traj['wing_states'][micro_step]
    #    xd_wing_state = xd[16:32, sec].copy()

    #    if not np.allclose(wing_state, xd_wing_state):
    #        print(f"Wing state at micro_step {micro_step} is not the same as the xd_wing_state at step {sec}")
    #        print(f"Wing state: {wing_state}")
    #        print(f"XD wing state: {xd_wing_state}")
    #        raise ValueError(f"Wing state at micro_step {micro_step} is not the same as the xd_wing_state at step {sec}")
    #    else:
    #        print(f"Wing state at micro_step {micro_step} is the same as the xd_wing_state at step {sec}")
    
    print(f"Simulating ghost trajectory from step {diff_step}...")
    lead_at_diff_step = lead_after_diff[0]
    wing_at_diff_step = wing_after_diff[0]


    ghost_traj = simulate_trajectory(diff_second, 10, use_nn_commands=True, 
                                     lead_init=lead_at_diff_step, wing_init=wing_at_diff_step)
    del ghost_traj['commands']
   
    # Calculate total frames
    total_frames = len(actual_traj['times']) + len(ghost_traj['times']) + 3*PAUSE_FRAMES  # normal + ghost + pause + avoidance
    print(f"ghost frames: {len(ghost_traj['times'])}, total_frames={total_frames}")
    
    # Setup figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Animation state variables
    anim_state = {
        'phase': 'normal',  # 'normal', 'ghost', 'command', 'avoidance'
        'current_step': 0,
        'ghost_step': None,
        'command_shown': False,
        'force_frame': None,
        'ghost_wing_trail': []   # Store ghost wing trail during ghost phase
    }
    
    # Text elements
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=14)
    distance_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, fontsize=14)
    command_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, fontsize=14)
    frame_text = ax.text2D(0.05, 0.05, "", transform=ax.transAxes, fontsize=12)

    print(f"First step where commands differ: {diff_step}")
    
    def animate(frame):
        """Animation function for each frame"""
        if anim_state['force_frame'] is not None:
            frame = anim_state['force_frame']

        print(f"Frame: {frame}/{total_frames}")
        
        ax.clear()

        zoom_frac = 0.0

        if frame > diff_step and frame <= diff_step + 2*PAUSE_FRAMES:
            zoom_frac = (frame - diff_step) / (2*PAUSE_FRAMES)
        elif frame > diff_step + 2*PAUSE_FRAMES:
            zoom_frac = 1.0

        ZOOMED_OUT_RADIUS = 4000
        ZOOMED_IN_RADIUS = 250
        zoom_radius = ZOOMED_OUT_RADIUS - (ZOOMED_OUT_RADIUS - ZOOMED_IN_RADIUS) * zoom_frac

        ZOOMED_OUT_ELEV = 45
        ZOOMED_IN_ELEV = 85
        view_elev = ZOOMED_OUT_ELEV - (ZOOMED_OUT_ELEV - ZOOMED_IN_ELEV) * zoom_frac

        ZOOMED_OUT_F16_SCALE = 60
        ZOOMED_IN_F16_SCALE = 1
        f16_scale = ZOOMED_OUT_F16_SCALE - (ZOOMED_OUT_F16_SCALE - ZOOMED_IN_F16_SCALE) * zoom_frac

        # Determine current phase and step
        if frame < diff_step:
            # Normal simulation phase
            anim_state['phase'] = 'normal'
            anim_state['current_step'] = frame
            anim_state['ghost_step'] = None

            print(f"Normal phase: {anim_state['current_step']}")
        elif frame < diff_step + 2*PAUSE_FRAMES:
            anim_state['phase'] = 'zoom'
            anim_state['current_step'] = diff_step
            anim_state['ghost_step'] = None

            

            print(f"Zoom phase: {anim_state['current_step']}, zoom_frac={zoom_frac}")
        elif frame < diff_step + 2*PAUSE_FRAMES + len(ghost_traj['times']):
            # Ghost prediction phase
            anim_state['phase'] = 'ghost'
            anim_state['ghost_step'] = frame - diff_step - 2*PAUSE_FRAMES
            anim_state['current_step'] = diff_step

            print(f"Ghost phase: {anim_state['ghost_step']}")
        elif frame < diff_step + len(ghost_traj['times']) + 3*PAUSE_FRAMES:  # PAUSE_FRAMES frames pause
            # Show command phase
            anim_state['phase'] = 'command'
            anim_state['current_step'] = diff_step
            anim_state['ghost_step'] = len(ghost_traj['times']) - 1

            print(f"Command phase: {anim_state['current_step']}")
        else:
            # Continue with avoidance
            anim_state['phase'] = 'avoidance'
            anim_state['current_step'] = frame - len(ghost_traj['times']) - 3*PAUSE_FRAMES
            anim_state['ghost_step'] = len(ghost_traj['times']) - 1

            print(f"Avoidance phase: {anim_state['current_step']}")
        
        # Get current states
        lead_state = actual_traj['lead_states'][anim_state['current_step']]
        wing_state = actual_traj['wing_states'][anim_state['current_step']]
        
        # Calculate positions
        lead_pos = np.array([lead_state[StateIndex.POS_E],
                           lead_state[StateIndex.POS_N],
                           lead_state[StateIndex.ALT]])
        wing_pos = np.array([wing_state[StateIndex.POS_E],
                           wing_state[StateIndex.POS_N],
                           wing_state[StateIndex.ALT]])
        
        # Draw aircraft

        # Show wingman in red during command display
        draw_f16_aircraft(ax, lead_state, color='black', scale=f16_scale)
        draw_f16_aircraft(ax, wing_state, color='black', scale=f16_scale)

        ghost_wing_pos = None

        if anim_state['ghost_step'] is not None:
            ghost_wing_pos = ghost_traj['wing_states'][anim_state['ghost_step']]
            if anim_state['phase'] in ['command', 'avoidance']:
                ghost_wing_color = 'red'
            else: 
                ghost_wing_color = 'grey'
            
            draw_f16_aircraft(ax, ghost_wing_pos, color=ghost_wing_color, alpha=GHOST_ALPHA, scale=f16_scale)
            
            # Save ghost wingman trail data during ghost phase
            if anim_state['phase'] in ['ghost','command']:
                ghost_wing_3pos = np.array([ghost_wing_pos[StateIndex.POS_E],
                                          ghost_wing_pos[StateIndex.POS_N],
                                          ghost_wing_pos[StateIndex.ALT]])
                
                # Append to trail if this is a new position
                if (len(anim_state['ghost_wing_trail']) == 0 or 
                    not np.array_equal(anim_state['ghost_wing_trail'][-1], ghost_wing_3pos)):
                    anim_state['ghost_wing_trail'].append(ghost_wing_3pos)
            
            # Draw ghost lead aircraft
            ghost_lead_pos = ghost_traj['lead_states'][anim_state['ghost_step']]
            draw_f16_aircraft(ax, ghost_lead_pos, color='grey', alpha=GHOST_ALPHA, scale=f16_scale)

            # Draw cylinder around lead aircraft during ghost, command, and avoidance phases
            if anim_state['phase'] in ['ghost','command', 'avoidance']:
                if anim_state['phase'] in ['ghost','command']:
                    ghost_3pos = np.array([ghost_lead_pos[StateIndex.POS_E],
                                ghost_lead_pos[StateIndex.POS_N],
                                ghost_lead_pos[StateIndex.ALT]])
                else:
                    ghost_3pos = np.array([lead_state[StateIndex.POS_E],
                                lead_state[StateIndex.POS_N],
                                lead_state[StateIndex.ALT]])
                
                draw_cylinder(ax, ghost_3pos, CYLINDER_RADIUS, CYLINDER_HEIGHT)
        
        
    
        # Draw trails
        current_step = anim_state['current_step']
        lead_trail = np.array([[s[StateIndex.POS_E], s[StateIndex.POS_N], s[StateIndex.ALT]] 
                              for s in actual_traj['lead_states'][:current_step+1]])
        wing_trail = np.array([[s[StateIndex.POS_E], s[StateIndex.POS_N], s[StateIndex.ALT]] 
                              for s in actual_traj['wing_states'][:current_step+1]])
        
        if len(lead_trail) > 1:
            ax.plot(lead_trail[:, 0], lead_trail[:, 1], lead_trail[:, 2], 
                   'k-', linewidth=2, alpha=0.5)
        if len(wing_trail) > 1:
            ax.plot(wing_trail[:, 0], wing_trail[:, 1], wing_trail[:, 2], 
                   'orange', linewidth=2, alpha=0.5)
            
        # Draw ghost trails during ghost, command and avoidance phases
        if anim_state['phase'] in ['ghost', 'command', 'avoidance'] and len(anim_state['ghost_wing_trail']) > 1:
            ghost_wing_trail = np.array(anim_state['ghost_wing_trail'])
            
            ax.plot(ghost_wing_trail[:, 0], ghost_wing_trail[:, 1], ghost_wing_trail[:, 2], 
                   color='gray', linewidth=2, alpha=0.8, linestyle=':')
        
        # Update text
        time_text.set_text(f'Time: {current_step / FPS:.1f}s')
        frame_text.set_text(f'Frame: {frame}')
        
        # Calculate and display distance
        distance = np.linalg.norm(lead_pos - wing_pos)
        distance_text.set_text(f'Distance: {distance:.1f}ft')
        
        # Show commands during command phase
        #if anim_state['phase'] == 'command':
        #    if current_step < len(actual_traj['commands']):
        #        ud_cmd = actual_traj['commands'][current_step]
        #        command_text.set_text(f'UD Command: Hdg={ud_cmd[0]:.3f}, Vel={ud_cmd[1]:.1f}')
        #else:
        #    command_text.set_text('')

        
        # Set axis limits centered on focus position
        #view_size = 400 if anim_state['phase'] in ['ghost', 'command'] else 1000  # Zoomed in by factor of 5
        #ax.set_xlim(focus_pos[0] - view_size, focus_pos[0] + view_size)
        #ax.set_ylim(focus_pos[1] - view_size, focus_pos[1] + view_size)
        #ax.set_zlim(focus_pos[2] - 100, focus_pos[2] + 100)  # Also zoom in vertically
      
        x = wing_state[StateIndex.POS_E]
        y = wing_state[StateIndex.POS_N]
        z = wing_state[StateIndex.ALT]

        if anim_state['phase'] in ['ghost', 'command']:
            x = ghost_wing_pos[StateIndex.POS_E]
            y = ghost_wing_pos[StateIndex.POS_N]
            z = ghost_wing_pos[StateIndex.ALT]

        ax.set_xlim([x - zoom_radius, x + zoom_radius])
        ax.set_ylim([y - zoom_radius, y + zoom_radius])
        ax.set_zlim([z - zoom_radius, z + zoom_radius])

        # Set view angle
        #ax.view_init(elev=45, azim=-60)
        psi = wing_state[StateIndex.PSI]
        ax.view_init(view_elev, -np.rad2deg(psi) - 90.0)
        
        # Labels
        ax.set_xlabel('East (ft)')
        ax.set_ylabel('North (ft)')
        ax.set_zlabel('Altitude (ft)')
#        ax.set_title('F-16 Wingman Scenario - Neural Network vs Commanded Trajectory')
        
        return []
    
    #print(f"DEBUG total frames = 90")
    #total_frames = 90
    
    # Create animation
    interval = 1 if LIVE_PLOT else 1000/FPS
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                 interval=interval, blit=False)  # 1ms interval for no delay

    if LIVE_PLOT:
        anim_state['force_frame'] = 0# 14 sec

        def on_key_press(event):
            """Handle keyboard input"""
            if event.key in ['right', 'd']:
                # Move forward one frame

                anim_state['force_frame'] += 10 if event.key == 'right' else 1

                if anim_state['force_frame'] > total_frames - 1:
                    anim_state['force_frame'] = 0

                #update_frame(new_frame)
                print(f"Right arrow key pressed. Frame: {anim_state['force_frame']}")
            elif event.key in ['left', 'a']:
                # Move backward one frame
                anim_state['force_frame'] -= 10 if event.key == 'left' else 1

                if anim_state['force_frame'] < 0:
                    anim_state['force_frame'] = total_frames - 1

                print(f"Left arrow key pressed. Frame: {anim_state['force_frame']}")

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        plt.show()
    else:
        # Save as video
        filename = 'f16_3d_animation.mp4'
        print(f"Saving animation to {filename}...")
        anim.save(filename, fps=FPS, writer='ffmpeg', dpi=100)
        print(f"Animation saved to {filename}")
    
    return anim

if __name__ == "__main__":
    create_animation() 
