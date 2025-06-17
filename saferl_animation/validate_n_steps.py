import numpy as np
import onnxruntime as ort
from scipy.io import loadmat
import sys
import os
import time
from scipy.integrate import RK45

sys.path.append('SafeRL')

from saferl.aerospace.models.f16.aerobench.example.wingman.wingman_autopilot import WingmanAutopilot
from saferl.aerospace.models.f16.aerobench.util import StateIndex, get_state_names, Euler, Freezable
from saferl.aerospace.models.f16.aerobench.highlevel.controlled_f16 import controlled_f16

print("=== N-Step Neural Network Validation (N=5) ===")

# Set N steps
N = 5 # Test for 5 full seconds

# Load the rollout data
m = loadmat('rollout.mat')
xd = m['xd']  # Shape: (38, 201) expected
dd = m['dd']  # Shape: (201, 1) expected
ud = m['ud']  # Shape: (201, 2) found

print(f"Loaded data shapes: xd={xd.shape}, dd={dd.shape}, ud={ud.shape}")

# Load the neural network
ort_session = ort.InferenceSession('ckpt_825.onnx')
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]

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

def vec2magnorm(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-6: return np.array([0, 0, 0])
    return np.concatenate(([norm], vec / norm))

def get_observation(lead_state_13dim, wing_state_13dim):
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
    obs = np.array([
        1.0, norm_rel_pos_lead[1], norm_rel_pos_lead[2],
        1.0, norm_rel_pos_rejoin[1], norm_rel_pos_rejoin[2],
        1.0, norm_wing_vel[1], norm_wing_vel[2],
        1.0, norm_lead_vel[1], norm_lead_vel[2]
    ])
    return obs

def get_network_action(obs):
    obs_exp = np.expand_dims(obs, axis=0).astype(np.float32)
    action = ort_session.run(output_names, {input_name: obs_exp})[0]
    heading_output = np.clip(action[0][0], -1, 1)
    velocity_output = np.clip(action[0][2], -1, 1)
    heading_change = heading_output * 0.174533
    velocity_change = velocity_output * 10.0
    return heading_change, velocity_change

initial_step_idx = 0
lead_initial_state = xd[:13, initial_step_idx].copy()
wing_initial_state = xd[16:29, initial_step_idx].copy()

print(f"\n=== Initial States (t={initial_step_idx}) ===")
print(f"Lead: Pos=({lead_initial_state[StateIndex.POS_E]:.1f}, {lead_initial_state[StateIndex.POS_N]:.1f}), Alt={lead_initial_state[StateIndex.ALT]:.0f}, Vel={lead_initial_state[StateIndex.VT]:.1f}, Hdg={lead_initial_state[StateIndex.PSI]:.3f}")
print(f"Wing: Pos=({wing_initial_state[StateIndex.POS_E]:.1f}, {wing_initial_state[StateIndex.POS_N]:.1f}), Alt={wing_initial_state[StateIndex.ALT]:.0f}, Vel={wing_initial_state[StateIndex.VT]:.1f}, Hdg={wing_initial_state[StateIndex.PSI]:.3f}")

current_lead_sim_state = lead_initial_state.copy()
current_wing_sim_state = wing_initial_state.copy()

# --- Lead Autopilot (Fixed Targets) & F16SimState ---
lead_ap_target_hdg_rad = lead_initial_state[StateIndex.PSI]
lead_ap_target_vel_fps = lead_initial_state[StateIndex.VT]
lead_ap_target_alt_ft = lead_initial_state[StateIndex.ALT]
lead_autopilot = WingmanAutopilot(
    target_heading=lead_ap_target_hdg_rad,
    target_vel=lead_ap_target_vel_fps,
    target_alt=lead_ap_target_alt_ft,
    stdout=False
)
lead_f16_sim = F16SimState(initial_state=lead_initial_state.copy(),
                           ap=lead_autopilot, 
                           step=1.0,
                           extended_states=False, 
                           integrator_str='rk45',
                           )
lead_f16_sim.init_simulation() 

print(f"Lead Autopilot: Target Hdg={lead_ap_target_hdg_rad:.4f} rad, Vel={lead_ap_target_vel_fps:.2f} ft/s, Alt={lead_ap_target_alt_ft:.2f} ft (Fixed)")
print(f"Lead F16SimState using integrator_kwargs: {lead_f16_sim.integrator_kwargs}")
# Compare with xd[13:16, 0]
if xd.shape[0] > 15:
    print(f"  rollout.mat xd[13:16, 0] (Potential Lead Targets?): {xd[13:16, initial_step_idx]}")

# --- Wingman Autopilot (Targets updated by NN) ---
wingman_ap_initial_target_hdg_rad = wing_initial_state[StateIndex.PSI]
wingman_ap_initial_target_vel_fps = wing_initial_state[StateIndex.VT]
wingman_ap_initial_target_alt_ft = wing_initial_state[StateIndex.ALT]
wingman_autopilot = WingmanAutopilot(
    target_heading=wingman_ap_initial_target_hdg_rad,
    target_vel=wingman_ap_initial_target_vel_fps,
    target_alt=wingman_ap_initial_target_alt_ft,
    stdout=False
)
# Instantiate F16SimState for Wingman - this will also use the modified integrator_kwargs
wingman_f16_sim = F16SimState(initial_state=wing_initial_state.copy(),
                              ap=wingman_autopilot, 
                              step=1.0, 
                              extended_states=False, 
                              integrator_str='rk45',
                              )
wingman_f16_sim.init_simulation()

v_min_wingman = 700.0
v_max_wingman = 900.0
print(f"Wingman Autopilot: Initial Target Hdg={wingman_ap_initial_target_hdg_rad:.4f} rad, Vel={wingman_ap_initial_target_vel_fps:.2f} ft/s, Alt={wingman_ap_initial_target_alt_ft:.2f} ft (Dynamic)")
print(f"Wingman F16SimState using integrator_kwargs: {wingman_f16_sim.integrator_kwargs}")
# Compare with xd[29:32, 0]
if xd.shape[0] > 31:
    print(f"  rollout.mat xd[29:32, 0] (Potential Wingman Initial Targets?): {xd[29:32, initial_step_idx]}")
# Compare with xd[32:38, 0]
if xd.shape[0] >= 38:
    print(f"  rollout.mat xd[32:38, 0] (Other Wingman-related?): {xd[32:38, initial_step_idx]}")
print(f"Wingman Velocity Limits: v_min={v_min_wingman:.1f} ft/s, v_max={v_max_wingman:.1f} ft/s (from config)")

print(f"\n=== Step-by-Step Validation (N={N} steps) ===")
overall_validation_passed = True
high_precision_tolerance = 0.1

for step_num in range(N):
    time_at_step_start = initial_step_idx + step_num
    
    print(f"\n--- Step {step_num}: t={time_at_step_start}s → t={time_at_step_start+1}s ---")

    # --- A. State Validation ---
    rec_lead_state_at_t = xd[:13, time_at_step_start]
    rec_wing_state_at_t = xd[16:29, time_at_step_start]

    lead_xy_sim_t = np.array([current_lead_sim_state[StateIndex.POS_E], current_lead_sim_state[StateIndex.POS_N]])
    lead_xy_rec_t = np.array([rec_lead_state_at_t[StateIndex.POS_E], rec_lead_state_at_t[StateIndex.POS_N]])
    lead_xy_norm_t = np.linalg.norm(lead_xy_sim_t - lead_xy_rec_t)
    
    wing_xy_sim_t = np.array([current_wing_sim_state[StateIndex.POS_E], current_wing_sim_state[StateIndex.POS_N]])
    wing_xy_rec_t = np.array([rec_wing_state_at_t[StateIndex.POS_E], rec_wing_state_at_t[StateIndex.POS_N]])
    wing_xy_norm_t = np.linalg.norm(wing_xy_sim_t - wing_xy_rec_t)
    
    input_pos_tolerance = 0.1 if step_num == 0 else 20.0
    input_state_match_lead = lead_xy_norm_t < (high_precision_tolerance if step_num > 0 else 0.1)
    input_state_match_wing = wing_xy_norm_t < input_pos_tolerance
    
    print(f"Input State: Lead XY diff={lead_xy_norm_t:.3f}ft {'✓' if input_state_match_lead else '✗'}, Wing XY diff={wing_xy_norm_t:.3f}ft {'✓' if input_state_match_wing else '✗'}")
    if not (input_state_match_lead and input_state_match_wing): overall_validation_passed = False
    
    # --- B. Action Validation ---
    current_obs_for_nn = get_observation(current_lead_sim_state, current_wing_sim_state)
    nn_cmd_hdg_chg, nn_cmd_vel_chg = get_network_action(current_obs_for_nn)
    rec_cmd_hdg_chg = ud[time_at_step_start, 0]
    rec_cmd_vel_chg = ud[time_at_step_start, 1]
    cmd_hdg_diff = abs(nn_cmd_hdg_chg - rec_cmd_hdg_chg)
    cmd_vel_diff = abs(nn_cmd_vel_chg - rec_cmd_vel_chg)
    
    cmd_match_hdg_tol = 0.01; cmd_match_vel_tol = 1.0
    command_match = cmd_hdg_diff < cmd_match_hdg_tol and cmd_vel_diff < cmd_match_vel_tol
    print(f"Command: Hdg diff={cmd_hdg_diff:.6f} {'✓' if cmd_hdg_diff < cmd_match_hdg_tol else '✗'}, Vel diff={cmd_vel_diff:.2f} {'✓' if cmd_vel_diff < cmd_match_vel_tol else '✗'}")
    if not command_match: overall_validation_passed = False

    # --- C. Update Wingman Autopilot Targets ---
    previous_target_hdg = wingman_autopilot.targets[0]
    wingman_autopilot.targets[0] -= nn_cmd_hdg_chg
    
    previous_target_vel = wingman_autopilot.targets[1]
    potential_new_target_vel = wingman_autopilot.targets[1] + nn_cmd_vel_chg
    if potential_new_target_vel < v_min_wingman: wingman_autopilot.targets[1] = v_min_wingman
    elif potential_new_target_vel > v_max_wingman: wingman_autopilot.targets[1] = v_max_wingman
    else: wingman_autopilot.targets[1] = potential_new_target_vel

    # Target Verification
    sim_targets_for_interval = wingman_autopilot.targets.copy()
    idx_for_recorded_targets = time_at_step_start + 1
    if idx_for_recorded_targets < xd.shape[1] and xd.shape[0] >= 38:
        recorded_targets_from_xd = xd[35:38, idx_for_recorded_targets]
        target_diff_norm = np.linalg.norm(sim_targets_for_interval - recorded_targets_from_xd)
        target_match = target_diff_norm < 1e-5
        print(f"Targets: L2 diff={target_diff_norm:.2e} {'✓' if target_match else '✗'}")

    # --- D. Simulate Lead Aircraft ---
    target_sim_time_lead = float(step_num + 1)
    try:
        lead_f16_sim.simulate_to(target_sim_time_lead, update_mode_at_start=True) 
        next_lead_sim_state_full = lead_f16_sim.states[-1] 
        next_lead_sim_state = next_lead_sim_state_full[:13].copy()
    except Exception as e:
        print(f"LEAD SIM FAILED: {e}")
        overall_validation_passed = False; break
        
    # --- E. Simulate Wingman Aircraft ---
    target_sim_time_wingman = float(step_num + 1)
    try:
        wingman_f16_sim.simulate_to(target_sim_time_wingman, update_mode_at_start=True)
        next_wing_sim_state_full = wingman_f16_sim.states[-1]
        next_wing_sim_state = next_wing_sim_state_full[:13].copy()
    except Exception as e:
        print(f"WINGMAN SIM FAILED: {e}")
        overall_validation_passed = False; break

    # --- F. Post-Simulation State Validation ---
    time_at_step_end = initial_step_idx + step_num + 1
    rec_lead_state_at_t_plus_1 = xd[:13, time_at_step_end]
    rec_wing_state_at_t_plus_1 = xd[16:29, time_at_step_end]

    lead_xy_sim_t_plus_1 = np.array([next_lead_sim_state[StateIndex.POS_E], next_lead_sim_state[StateIndex.POS_N]])
    lead_xy_rec_t_plus_1 = np.array([rec_lead_state_at_t_plus_1[StateIndex.POS_E], rec_lead_state_at_t_plus_1[StateIndex.POS_N]])
    lead_xy_norm_t_plus_1 = np.linalg.norm(lead_xy_sim_t_plus_1 - lead_xy_rec_t_plus_1)

    wing_xy_sim_t_plus_1 = np.array([next_wing_sim_state[StateIndex.POS_E], next_wing_sim_state[StateIndex.POS_N]])
    wing_xy_rec_t_plus_1 = np.array([rec_wing_state_at_t_plus_1[StateIndex.POS_E], rec_wing_state_at_t_plus_1[StateIndex.POS_N]])
    wing_xy_norm_t_plus_1 = np.linalg.norm(wing_xy_sim_t_plus_1 - wing_xy_rec_t_plus_1)
    
    # Key differences for monitoring
    alt_diff_wing = abs(next_wing_sim_state[StateIndex.ALT] - rec_wing_state_at_t_plus_1[StateIndex.ALT])
    phi_diff_wing = abs(next_wing_sim_state[StateIndex.PHI] - rec_wing_state_at_t_plus_1[StateIndex.PHI])
    
    output_state_match_lead = lead_xy_norm_t_plus_1 < high_precision_tolerance
    output_state_match_wing = wing_xy_norm_t_plus_1 < high_precision_tolerance
    
    print(f"Output: Lead XY diff={lead_xy_norm_t_plus_1:.3f}ft {'✓' if output_state_match_lead else '✗'}, Wing XY diff={wing_xy_norm_t_plus_1:.3f}ft {'✓' if output_state_match_wing else '✗'}")
    print(f"Wing: Alt diff={alt_diff_wing:.1f}ft, Roll diff={phi_diff_wing:.4f}rad")
    
    if not (output_state_match_lead and output_state_match_wing):
        if not output_state_match_wing or (not output_state_match_lead and lead_xy_norm_t_plus_1 > 1.0):
             overall_validation_passed = False

    current_lead_sim_state = next_lead_sim_state
    current_wing_sim_state = next_wing_sim_state

print(f"\n=== VALIDATION RESULT ===")
if overall_validation_passed:
    print(f"✅ N-STEP VALIDATION PASSED for {N} steps!")
    print("Both lead and wingman aircraft simulations match recorded data within tolerances.")
else:
    print(f"❌ VALIDATION FAILED!")
    print("Some validation checks failed - see details above.")

print(f"=== Script Complete ===") 
