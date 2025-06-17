import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import onnxruntime as ort
from scipy.io import loadmat
import sys
from PIL import Image
import time
import csv
from scipy.integrate import RK45

# Add the correct SafeRL path instead of AeroBench
sys.path.append('SafeRL')

from saferl.aerospace.models.f16.aerobench.example.wingman.wingman_autopilot import WingmanAutopilot
from saferl.aerospace.models.f16.aerobench.util import StateIndex, get_state_names, Euler, Freezable
from saferl.aerospace.models.f16.aerobench.highlevel.controlled_f16 import controlled_f16

# Load the rollout data
m = loadmat('RTA_825_100.mat')
xd = m['xd'].T  # Shape: (38, 201) expected
dd = m['dd']  # Shape: (201, 1) expected
udm = m['udm']  # Shape: (201, 2) found
#ad = m['ad']  # Additional field
#cmdd = m['cmdd']  # Command mode (1 or 2)
obsd = m['obsd']  # Shape: (201, 12) expected
td = m['td']  # Time data
actionsd = m['actionsd']  # Use this for tolerance checking instead of ud
# Load the csv file

csv_file_path = 'eval_csvfile.csv'

#t,name,control[0],control[1],ap.targets[0],ap.targets[1],ap.targets[2],f16_state[0],f16_state[1],f16_state[2],f16_state[3],f16_state[4],f16_state[5],f16_state[6],f16_state[7],f16_state[8],f16_state[9],f16_state[10],f16_state[11],f16_state[12]

csv_wingman_states = []
csv_lead_states = []

with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'wingman':
            csv_wingman_states.append(np.array([float(x) for x in row[2:]]))
        elif row[1] == 'lead':
            csv_lead_states.append(np.array([float(x) for x in row[2:]]))

print(f"Loaded {len(csv_wingman_states)} wingman states from csv file.")

# Load the neural network
ort_session = ort.InferenceSession('ckpt_825.onnx')
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]

# Limit to first N seconds instead of full data
N_STEPS = 100
n_steps = min(N_STEPS, xd.shape[1])
print(f"Using first {n_steps} seconds of data")

airplane_img = Image.open('SafeRL/saferl/aerospace/models/f16/aerobench/visualize/airplane.png')

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
    """Convert 2D vector to magnitude-normalized form [mag, x/norm, y/norm]"""
    norm = np.linalg.norm(vec)
    if norm < 1e-6: 
        return np.array([0, 0, 0])
    return np.concatenate(([norm], vec / norm))

def get_observation(lead_state_13dim, wing_state_13dim):
    """Convert state vectors to normalized observation using magnorm mode - correct version from validate_n_steps.py"""
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

def get_color(step, TOL=1e-5):
    """Get color based on neural network validation vs actual commands"""
    if step >= udm.shape[0] or step >= n_steps:
        return 'black'  # Default for out of range
    
    # Get current combined state (lead + wingman)
    lead_state = xd[:13, step]
    wing_state = xd[16:29, step]
    
    # Get observation and neural network action using correct observation function
    obs = get_observation(lead_state, wing_state)
    nn_heading, nn_velocity = get_network_action(obs)
    
    # Get actual recorded actions from different sources

    udm_heading = udm[step, 0]
    udm_velocity = udm[step, 1]
    
    # Check if they match within tolerance
    nn_udm_heading_match = abs(nn_heading - udm_heading) < TOL
    nn_udm_velocity_match = abs(nn_velocity - udm_velocity) < TOL
    
    # Determine color based on matches
    if nn_udm_heading_match and nn_udm_velocity_match:
        return 'black'  # All match
    else:
        return 'red'

def add_airplane_markers(ax, x, y, heading, color='blue', alpha=1.0, size=None):
    """Add airplane marker at given position with heading, size auto if None"""
    
    markers = []

    if False:
        # 1. Draw the quiver arrow (underneath)
        
        frac = 0.1
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        arrow_len = frac * np.hypot(xspan, yspan)
        u = arrow_len * np.cos(heading)
        v = arrow_len * np.sin(heading)

        q = ax.quiver(x, y, u, v,
                      angles='xy', scale_units='xy', scale=1,
                      width=0.01, headwidth=4, headlength=6,
                      color=color, alpha=alpha, zorder=10)
        markers.append(q)

    # 2. Overlay airplane image 
    
    # --- Prepare the rotated image ---
    angle_deg = np.degrees(heading) - 90
    img = airplane_img.rotate(angle_deg, expand=False)

    # Optionally tint for color (very basic: only red/black, can be improved)
    img_array = np.array(img.convert('RGBA'))
    
    # Identify non-transparent (i.e., plane pixels)
    mask = img_array[..., 3] > 0
    
    if color == 'red':
        img_array[..., 0][mask] = 220   # Red
        img_array[..., 1][mask] = 50    # Green (low)
        img_array[..., 2][mask] = 50    # Blue (low)
    elif color == 'blue':
        img_array[..., 0][mask] = 50
        img_array[..., 1][mask] = 50
        img_array[..., 2][mask] = 220
    elif color == 'orange':
        img_array[..., 0][mask] = 255   # Red
        img_array[..., 1][mask] = 165   # Green
        img_array[..., 2][mask] = 0     # Blue
    
    img_array[..., 3] = (img_array[..., 3] * alpha).astype(np.uint8)   # apply alpha
    img = Image.fromarray(img_array)

    # --- Figure out image extent (size in data coords) ---
    icon_span = 1200
    extent = [x - icon_span/2, x + icon_span/2,
              y - icon_span/2, y + icon_span/2]

    im = ax.imshow(img, extent=extent, zorder=11, alpha=alpha)
    markers.append(im)

    return markers

  

# Compute the lead and wingman positions
lead_positions = xd[[StateIndex.POS_E, StateIndex.POS_N], :n_steps].T  # Shape: (n_steps, 2)
wing_positions = xd[[StateIndex.POS_E + 16, StateIndex.POS_N + 16], :n_steps].T  # Shape: (n_steps, 2)
lead_headings = np.pi/2 - xd[StateIndex.PSI, :n_steps]  # Convert PSI to standard heading
wing_headings = np.pi/2 - xd[StateIndex.PSI + 16, :n_steps]  # Convert PSI to standard heading

# Storage for rollout data
ROLLOUT_SECS = 25
rollout_data = None
rollout_step = 0
prediction_slider = None  # Second slider for navigating through predictions
prediction_step = 0  # Current step in the prediction

def compute_nn_rollout(initial_step, duration=10):
    """Compute neural network rollout starting from initial_step for duration seconds using correct simulation from validate_n_steps.py"""
    global rollout_data, rollout_step
    
    print(f"Computing {duration}-second rollout from step {initial_step}...")
    
    # Get initial states (was 13, now 16)
    lead_initial_state = xd[:16, initial_step].copy()
    wing_initial_state = xd[16:32, initial_step].copy()

    #wing_initial_state = csv_wingman_states[initial_step][-16:].copy()
    #lead_initial_state = csv_lead_states[initial_step][-16:].copy()

    assert len(lead_initial_state) == 16, f"lead_initial_state has {len(lead_initial_state)} elements, expected 16"
    assert len(wing_initial_state) == 16, f"wing_initial_state has {len(wing_initial_state)} elements, expected 16"
    
    # Setup lead autopilot with fixed targets
    lead_ap_target_hdg_rad = xd[StateIndex.PSI, 0]  # Initial PSI at step 0
    lead_ap_target_vel_fps = xd[StateIndex.VT, 0]   # Initial VT at step 0
    lead_ap_target_alt_ft = xd[StateIndex.ALT, 0]   # Initial ALT at step 0
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
                               integrator_str='rk45')
    lead_f16_sim.init_simulation()
    
    if initial_step > 0:
        # get the wingman targets from the xd file
        wingman_ap_target_hdg_rad = xd[35, initial_step]
        wingman_ap_target_vel_fps = xd[36, initial_step]                                                                                
        wingman_ap_target_alt_ft = xd[37, initial_step]
    else:
        # Compute correct wingman autopilot targets by starting from step 0 and applying commands
        # targets at step 0 are the state values
        wingman_ap_target_hdg_rad = xd[16 + StateIndex.PSI, 0]  # Initial PSI at step 0
        wingman_ap_target_vel_fps = xd[16 + StateIndex.VT, 0]   # Initial VT at step 0
        wingman_ap_target_alt_ft = xd[16 + StateIndex.ALT, 0]   # Initial ALT at step 0
    
    # Setup wingman autopilot with computed targets
    wingman_autopilot = WingmanAutopilot(
        target_heading=wingman_ap_target_hdg_rad,
        target_vel=wingman_ap_target_vel_fps,
        target_alt=wingman_ap_target_alt_ft,
        stdout=False
    )
    wingman_f16_sim = F16SimState(initial_state=wing_initial_state.copy(),
                                  ap=wingman_autopilot, 
                                  step=1.0, 
                                  extended_states=False, 
                                  integrator_str='rk45')
    wingman_f16_sim.init_simulation()
    
    # Simulate step by step
    lead_positions = []
    wing_positions = []
    wing_headings = []
    lead_full_states = []
    wing_full_states = []
    times = []
    
    current_lead_sim_state = lead_initial_state.copy()
    current_wing_sim_state = wing_initial_state.copy()
    
    # store initial positions at t=0
    lead_positions.append([current_lead_sim_state[StateIndex.POS_E], current_lead_sim_state[StateIndex.POS_N]])
    wing_positions.append([current_wing_sim_state[StateIndex.POS_E], current_wing_sim_state[StateIndex.POS_N]])
    wing_headings.append(np.pi/2 - current_wing_sim_state[StateIndex.PSI])
    lead_full_states.append(lead_initial_state.copy())
    wing_full_states.append(wing_initial_state.copy())
    times.append(0.0)

    for step in range(1, duration+1):
        # Update progress
        progress = int(((step-1) / duration) * 100)
        button_rollout.label.set_text(f'Computing... {progress}%')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Get neural network command
        current_obs_for_nn = get_observation(current_lead_sim_state, current_wing_sim_state)
        nn_cmd_hdg_chg, nn_cmd_vel_chg = get_network_action(current_obs_for_nn)

        if initial_step in [0, 1]:     
            step_index = initial_step + step - 1

            sim_f16_state = wingman_f16_sim.states[-1]
            assert len(sim_f16_state) == 16, f"sim_f16_state has {len(sim_f16_state)} elements, expected 16"

            sim_targets = np.array(wingman_autopilot.targets)
            sim_control = np.array([nn_cmd_hdg_chg, nn_cmd_vel_chg])

            if step_index < len(csv_wingman_states):
                # print differences between csv, sim, and xd(matlab)
                csv_f16_control = csv_wingman_states[step_index][0:2]
                csv_f16_targets = csv_wingman_states[step_index][2:5]
                csv_f16_state = csv_wingman_states[step_index][5:]

                csv_sim_diff = csv_f16_state - sim_f16_state
                if np.linalg.norm(csv_sim_diff) == 0:
                    print(f"\nstep {step_index}: csv_sim_diff={np.linalg.norm(csv_sim_diff):.17g}")
                else:
                    print(f"\nstep {step_index}: csv_sim_diff={np.linalg.norm(csv_sim_diff):.17g} - {[f'{x:.17g}' for x in csv_sim_diff]}")

                csv_sim_diff_control = repr(csv_f16_control - sim_control)
                csv_sim_diff_targets = repr(csv_f16_targets - sim_targets)
                print(f"step {step_index}: {csv_sim_diff_control=}")
                print(f"step {step_index}: {csv_sim_diff_targets=}")

            #### do the same thing, but now using xd data
            if step_index < xd.shape[1]:
                if step_index > 0:
                    xd_f16_targets = xd[35:38, step_index] # was -1
                else:
                    t0 = xd[16 + StateIndex.PSI, 0]  # Initial PSI at step 0
                    t1 = xd[16 + StateIndex.VT, 0]   # Initial VT at step 0
                    t2 = xd[16 + StateIndex.ALT, 0]   # Initial ALT at step 0
                    xd_f16_targets = np.array([t0, t1, t2])

                xd_f16_state = xd[16:32, step_index] # was -1

                xd_sim_diff = xd_f16_state - sim_f16_state
                if np.linalg.norm(xd_sim_diff) == 0:
                    print(f"\nstep {step_index}: xd_sim_diff={np.linalg.norm(xd_sim_diff):.17g}")
                else:
                    print(f"\nstep {step_index}: xd_sim_diff={np.linalg.norm(xd_sim_diff):.17g} - {[f'{x:.17g}' for x in xd_sim_diff]}")

                xd_sim_diff_targets = xd_f16_targets - sim_targets
                print(f"step {step_index}: {xd_sim_diff_targets=}")

                # for controls do ud and udm
                ud_f16_control = udm[step_index, 0:2]
                ud_sim_diff_control = repr(ud_f16_control - sim_control)
                print(f"step {step_index}: {ud_sim_diff_control=}")

                udm_f16_control = udm[step_index]
                udm_sim_diff_control = repr(udm_f16_control - sim_control)
                print(f"step {step_index}: {udm_sim_diff_control=}")
        
        # Update wingman autopilot targets
        wingman_autopilot.targets[0] -= nn_cmd_hdg_chg

        v_min_wingman = 700.0
        v_max_wingman = 900.0

        potential_new_target_vel = wingman_autopilot.targets[1] + nn_cmd_vel_chg
        if potential_new_target_vel < v_min_wingman: 
            wingman_autopilot.targets[1] = v_min_wingman
        elif potential_new_target_vel > v_max_wingman: 
            wingman_autopilot.targets[1] = v_max_wingman
        else: 
            wingman_autopilot.targets[1] = potential_new_target_vel
        
        # Simulate both aircraft for 1 second
        target_sim_time = float(step)
        
        lead_f16_sim.simulate_to(target_sim_time, update_mode_at_start=True)
        next_lead_sim_state = lead_f16_sim.states[-1][:16].copy()
        
        wingman_f16_sim.simulate_to(target_sim_time, update_mode_at_start=True)
        next_wing_sim_state = wingman_f16_sim.states[-1][:16].copy()
        
        # Store results
        lead_full_states.append(next_lead_sim_state)
        wing_full_states.append(next_wing_sim_state)
        lead_positions.append([next_lead_sim_state[StateIndex.POS_E], next_lead_sim_state[StateIndex.POS_N]])
        wing_positions.append([next_wing_sim_state[StateIndex.POS_E], next_wing_sim_state[StateIndex.POS_N]])
        wing_headings.append(np.pi/2 - next_wing_sim_state[StateIndex.PSI])
        times.append(target_sim_time)
        
        # Update current states for next iteration
        current_lead_sim_state = next_lead_sim_state
        current_wing_sim_state = next_wing_sim_state
    
    print(f"Rollout computed successfully, {len(times)} time points")
    
    rollout_data = {
        'lead_positions': np.array(lead_positions),
        'wing_positions': np.array(wing_positions), 
        'wing_headings': np.array(wing_headings),
        'times': np.array(times),
        'lead_full_states': np.array(lead_full_states),
        'wing_full_states': np.array(wing_full_states),
        'initial_step': initial_step
    }
    rollout_step = initial_step


def update_plot(step):
    """Update the plot for the given step"""
    ax.clear()
    
    # Plot full trajectories using dotted lines to prevent scale changes during animation
    ax.plot(lead_positions[:, 0], lead_positions[:, 1], ':', alpha=0.5, 
            color='black', linewidth=1, label='Lead trail (full)')
    ax.plot(wing_positions[:, 0], wing_positions[:, 1], ':', alpha=0.5, 
            color='orange', linewidth=1, label='Wingman trail (full)')
    
    # Plot trails up to current step with solid lines
    ax.plot(lead_positions[:step+1, 0], lead_positions[:step+1, 1], '-', 
            color='black', linewidth=2, label='Lead trail')
    ax.plot(wing_positions[:step+1, 0], wing_positions[:step+1, 1], '-', 
            color='orange', linewidth=2, label='Wingman trail')
    
    # Plot dot at initial position (step 0)
    ax.plot(lead_positions[0, 0], lead_positions[0, 1], 'o', 
            color='black', markersize=8, label='Lead start')
    ax.plot(wing_positions[0, 0], wing_positions[0, 1], 'o', 
            color='orange', markersize=8, label='Wingman start')
    
    # Storage for airplane markers (to return for cleanup)
    airplane_markers = []
    
    # Add airplane markers at current position
    # Lead aircraft (blue)
    airplane_markers += add_airplane_markers(ax, lead_positions[step, 0], lead_positions[step, 1], 
                                               np.pi/2 - xd[StateIndex.PSI, step], color='black', size=None)
    
    # Wingman aircraft with color based on command validation
    TOL = 1e-5
    wing_color = get_color(step, TOL)
    airplane_markers += add_airplane_markers(ax, wing_positions[step, 0], wing_positions[step, 1], 
                                               np.pi/2 - xd[StateIndex.PSI + 16, step], color=wing_color, size=None)
    
    # Plot rollout data if available and relevant
    if rollout_data is not None and rollout_step == step:
        # Show rollout prediction as ghost aircraft with dotted trajectory
        rollout_lead_pos = rollout_data['lead_positions']
        rollout_wing_pos = rollout_data['wing_positions']
        rollout_wing_head = rollout_data['wing_headings']
        
        # Plot rollout trajectories with dotted lines
        ax.plot(rollout_lead_pos[:, 0], rollout_lead_pos[:, 1], ':', 
                color='lightblue', linewidth=2, alpha=0.7, label='Lead prediction')
        ax.plot(rollout_wing_pos[:, 0], rollout_wing_pos[:, 1], ':', 
                color='lightcoral', linewidth=2, alpha=0.7, label='Wingman prediction')
        
        # Show prediction at current prediction_step
        pred_idx = min(prediction_step, len(rollout_lead_pos) - 1)
        airplane_markers += add_airplane_markers(ax, rollout_lead_pos[pred_idx, 0], 
                                                   rollout_lead_pos[pred_idx, 1], 
                                                   np.pi/2 - xd[StateIndex.PSI, step], color='lightblue', 
                                                   alpha=0.7, size=None)
        airplane_markers += add_airplane_markers(ax, rollout_wing_pos[pred_idx, 0], 
                                                   rollout_wing_pos[pred_idx, 1], 
                                                   rollout_wing_head[pred_idx], color='lightcoral', 
                                                   alpha=0.7, size=None)
        
        # Draw dashed line between predicted aircraft positions with distance
        pred_distance = np.linalg.norm(rollout_lead_pos[pred_idx] - rollout_wing_pos[pred_idx])
        ax.plot([rollout_lead_pos[pred_idx, 0], rollout_wing_pos[pred_idx, 0]], 
                [rollout_lead_pos[pred_idx, 1], rollout_wing_pos[pred_idx, 1]], 
                '--', color='gray', alpha=0.7, linewidth=1)
        
        # print to terminal the distance between the wingman prediction and the wingman position in xd
        if step+pred_idx < xd.shape[1]:
            xd_wingman_f16_state = xd[16:32, step+pred_idx]
            xd_lead_f16_state = xd[0:16, step+pred_idx]

            xd_xy_wingman = np.array([xd_wingman_f16_state[StateIndex.POS_E], xd_wingman_f16_state[StateIndex.POS_N]])
            xd_xy_lead = np.array([xd_lead_f16_state[StateIndex.POS_E], xd_lead_f16_state[StateIndex.POS_N]])
            print(f"ghost step {pred_idx}: wingman x-y diff={np.linalg.norm(rollout_wing_pos[pred_idx] - xd_xy_wingman)}")
            print(f"ghost step {pred_idx}: lead x-y diff={np.linalg.norm(rollout_lead_pos[pred_idx] - xd_xy_lead)}")

            # more detailed comparison of full states
            rollout_wing_full_state = rollout_data['wing_full_states'][pred_idx]
            rollout_lead_full_state = rollout_data['lead_full_states'][pred_idx]
            # compare difference element by element
            rollout_wing_full_state_diff = rollout_wing_full_state - xd_wingman_f16_state
            rollout_lead_full_state_diff = rollout_lead_full_state - xd_lead_f16_state
            print(f"\nghost step {pred_idx}: wingman full state diff={np.linalg.norm(rollout_wing_full_state_diff)}")
            print(f"ghost step {pred_idx}: wingman full state diff={rollout_wing_full_state_diff}")

            print(f"\nghost step {pred_idx}: lead full state diff={np.linalg.norm(rollout_lead_full_state_diff)}")
            print(f"ghost step {pred_idx}: lead full state diff={rollout_lead_full_state_diff}")

    
    # Compute and display distance between aircraft
    current_distance = np.linalg.norm(lead_positions[step] - wing_positions[step])

    
    
    # Draw dashed line between current aircraft
    ax.plot([lead_positions[step, 0], wing_positions[step, 0]], 
            [lead_positions[step, 1], wing_positions[step, 1]], 
            '--', color='gray', linewidth=1)
    
    
    info_kwargs = dict(ha='left', va='top',
                   fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
    
    # Current distance label (top-left)
    ax.text(0.02, 0.98,
        f'Current Distance: {current_distance:.0f} ft',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='yellow', alpha=0.9),
        **info_kwargs)
    
    # Get NN command and display with color coding
    if step < udm.shape[0] and step < n_steps:
        # Get current combined state (lead + wingman)
        lead_state = xd[:13, step]
        wing_state = xd[16:29, step]
        
        # Get observation and neural network action
        obs = get_observation(lead_state, wing_state)
        nn_heading, nn_velocity = get_network_action(obs)
        
        # Get actual recorded actions from all sources
        ud_heading = udm[step, 0]
        ud_velocity = udm[step, 1]
        
        # Get command mode
        cmd_mode = 'TODO'
        
        # Check matches for color coding
        nn_ud_heading_match = abs(nn_heading - ud_heading) <= TOL
        nn_ud_velocity_match = abs(nn_velocity - ud_velocity) <= TOL
        
        # Determine color based on matches
        if nn_ud_heading_match and nn_ud_velocity_match:
            command_color = 'lightgreen'  # All match
        else:
            command_color = 'red'  # All three different
        
        # Display all three commands with color coding
        ax.text(0.02, 0.88,
            f'NN Cmd: Hdg={nn_heading:.3f}, Vel={nn_velocity:.1f}\n' + \
            f'UD Cmd: Hdg={ud_heading:.3f}, Vel={ud_velocity:.1f}\n' + \
            f'Cmd Mode: {cmd_mode}',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=command_color, alpha=0.9),
            **info_kwargs)

        # print raw input (obs) comparison as well as uad (raw output) comparison
        raw_heading, raw_velocity = get_network_action(obs, output_scaling=False)
        nn_actionsd = np.array([raw_heading, raw_velocity])

        #print(f" {obs=} ")

        if np.linalg.norm(obs - obsd[step]) > TOL:
            #print(f" {obs=}")
            print(f" OBS MISMATCH at step {step}: {np.linalg.norm(obs - obsd[step])}")
            print(f" {obs=} ")
            print(f" {obsd[step]=}")

            

            #print(f" {nn_actionsd=} {actionsd[step]=}")
            
        # if color is not green, print raw data to full precision
        if command_color != 'lightgreen':
            print(f" {command_color=}")

            # print raw input (obs) comparison as well as uad (raw output) comparison
            raw_heading, raw_velocity = get_network_action(obs, output_scaling=False)
            nn_actionsd = np.array([raw_heading, raw_velocity])
            print(f"\n {obs=} {obsd[step]=}")
            print(f" {nn_actionsd=} {actionsd[step]=}")

            if not nn_ud_heading_match:
                print(f" {nn_heading=} {ud_heading=} {ud_heading=}")

            if not nn_ud_velocity_match:
                print(f" {nn_velocity=} {ud_velocity=} {ud_velocity=}")

        
    
    # Predicted distance label (if rollout exists)
    if rollout_data is not None and rollout_step == step:
        rollout_lead_pos = rollout_data['lead_positions']
        rollout_wing_pos = rollout_data['wing_positions']
        pred_idx = min(prediction_step, len(rollout_lead_pos) - 1)
        distance = np.linalg.norm(rollout_lead_pos[pred_idx] - rollout_wing_pos[pred_idx])
        
        ax.text(0.02, 0.73,
            f'Predicted Distance (+{pred_idx}s): {distance:.0f} ft',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='lightcoral', alpha=0.9),
            **info_kwargs)
    
    # Set up the plot
    ax.set_xlabel('East Position (ft)')
    ax.set_ylabel('North Position (ft)')
    ax.set_title(f'F-16 Neural Network Validation - Step {step}/{n_steps-1} (Time: {step:.1f}s)')
    ax.grid(True, alpha=0.5)
    
    lx, ly = lead_positions[step]          # lead's current position
    lead_heading = lead_headings[step]
    
    wx = lx + 5000 * np.cos(lead_heading)
    wy = ly + 5000 * np.sin(lead_heading)
    
    HALF_X = 20000
    HALF_Y = 10000
    ax.set_xlim(wx - HALF_X, wx + HALF_X)
    ax.set_ylim(wy - HALF_Y, wy + HALF_Y)
    ax.set_aspect('equal', adjustable='box')   # keep axes perfectly square
    
    # Position legend in lower right
    ax.legend(loc='lower right', fontsize=12)
    
    # Store for cleanup
    return airplane_markers

def create_prediction_slider():
    """Create the prediction slider when rollout data is available"""
    global prediction_slider
    if rollout_data is not None and prediction_slider is None:
        ax_pred_slider = plt.axes([0.1, 0.05, 0.5, 0.04])  # Shortened slider width
        max_pred_steps = len(rollout_data['lead_positions']) - 1
        prediction_slider = Slider(ax_pred_slider, 'Ghost Step', 0, max_pred_steps, 
                                 valinit=0, valstep=1, valfmt='%d')
        prediction_slider.on_changed(on_prediction_slider_change)
        
        # Add ghost +/- buttons
        ax_ghost_minus = plt.axes([0.66, 0.05, 0.03, 0.045])
        ax_ghost_plus = plt.axes([0.70, 0.05, 0.03, 0.045])
        
        global button_ghost_minus, button_ghost_plus
        button_ghost_minus = Button(ax_ghost_minus, '-')
        button_ghost_plus = Button(ax_ghost_plus, '+')
        
        button_ghost_minus.on_clicked(on_ghost_minus_button_click)
        button_ghost_plus.on_clicked(on_ghost_plus_button_click)
        
def remove_prediction_slider():
    """Remove the prediction slider"""
    global prediction_slider, prediction_step, button_ghost_minus, button_ghost_plus
    if prediction_slider is not None:
        prediction_slider.ax.remove()
        prediction_slider = None
        prediction_step = 0
        
    # Remove ghost buttons if they exist
    if 'button_ghost_minus' in globals() and button_ghost_minus is not None:
        button_ghost_minus.ax.remove()
        button_ghost_minus = None
    if 'button_ghost_plus' in globals() and button_ghost_plus is not None:
        button_ghost_plus.ax.remove()
        button_ghost_plus = None

def on_prediction_slider_change(val):
    """Handle prediction slider value change"""
    global prediction_step
    prediction_step = int(prediction_slider.val)
    airplane_markers = update_plot(int(slider.val))
    fig.canvas.draw()

def on_slider_change(val):
    """Handle slider value change"""
    global rollout_data, rollout_step, prediction_step
    step = int(slider.val)
    
    # Clear rollout data if slider moved (since rollout is only valid for the step it was computed)
    if rollout_data is not None and rollout_step != step:
        rollout_data = None
        remove_prediction_slider()
    
    airplane_markers = update_plot(step)
    fig.canvas.draw()

def on_rollout_button_click(event):
    """Handle rollout button click"""
    global prediction_step
    current_step = int(slider.val)
    
    # Update button to show progress
    button_rollout.label.set_text('Computing... 0%')
    fig.canvas.draw()
    
    # Remove existing prediction slider
    remove_prediction_slider()
    prediction_step = 0
    
    # Compute rollout
    compute_nn_rollout(current_step, duration=ROLLOUT_SECS)
    
    # Update button back to normal
    button_rollout.label.set_text(f'Compute {ROLLOUT_SECS}s Rollout')
    
    # Create prediction slider
    create_prediction_slider()
    
    # Refresh plot
    airplane_markers = update_plot(current_step)
    fig.canvas.draw()

def on_ghost_minus_button_click(event):
    """Handle ghost minus button click - decrement ghost step by 1"""
    if prediction_slider is not None:
        current_val = int(prediction_slider.val)
        if current_val > 0:
            prediction_slider.set_val(current_val - 1)

def on_ghost_plus_button_click(event):
    """Handle ghost plus button click - increment ghost step by 1"""
    if prediction_slider is not None:
        current_val = int(prediction_slider.val)
        max_val = len(rollout_data['lead_positions']) - 1 if rollout_data is not None else 0
        if current_val < max_val:
            prediction_slider.set_val(current_val + 1)

def on_minus_button_click(event):
    """Handle minus button click - decrement time step by 1"""
    current_val = int(slider.val)
    if current_val > 0:
        slider.set_val(current_val - 1)

def on_plus_button_click(event):
    """Handle plus button click - increment time step by 1"""
    current_val = int(slider.val)
    if current_val < n_steps - 1:
        slider.set_val(current_val + 1)

# Create the plot - use all available space
fig, ax = plt.subplots(figsize=(12, 8))  # Made wider from (10, 8) to (12, 8)
plt.subplots_adjust(bottom=0.25)

# Update slider axes positions - adjusted layout for new buttons
ax_slider = plt.axes([0.1, 0.11, 0.5, 0.04])  # Shortened slider width
slider = Slider(ax_slider, 'Time Step', 0, n_steps-1, valstep=1, valinit=0, valfmt='%d')
slider.on_changed(on_slider_change)

# Add - button
ax_button_minus = plt.axes([0.66, 0.11, 0.03, 0.045])
button_minus = Button(ax_button_minus, '-')

# Add + button  
ax_button_plus = plt.axes([0.70, 0.11, 0.03, 0.045])
button_plus = Button(ax_button_plus, '+')

# Rollout button - moved and made wider
ax_button = plt.axes([0.75, 0.11, 0.2, 0.045])
button_rollout = Button(ax_button, f'Compute {ROLLOUT_SECS}s Rollout')
button_rollout.on_clicked(on_rollout_button_click)

# Connect button callbacks
button_minus.on_clicked(on_minus_button_click)
button_plus.on_clicked(on_plus_button_click)

# Initial plot
airplane_markers = update_plot(0)

# auto-compute rollout at start
#compute_nn_rollout(0, duration=ROLLOUT_SECS)

plt.show()
