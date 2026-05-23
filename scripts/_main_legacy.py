import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import sys
import utils
import antennaeArray
import data
import solver
import os
from tqdm import tqdm
from matplotlib.patches import Circle
import pickle
import h5py
import scipy
import psutil

# --- Plotting Helper Functions ---

def plot_performance_vs_gmin(gmin_values, max_reflected_powers, max_array_gains_per_segment, baseline_power, baseline_gain_per_segment, case_number):
    """
    Plots the maximum reflected power and array gain as a function of the gmin parameter for a specific case.
    The array gain plot will show a separate curve for each antenna segment.
    """
    # Plot for Reflected Power
    plt.figure(figsize=(10, 6))
    plt.semilogx(gmin_values, max_reflected_powers, marker='o', linestyle='-', label='Optimized Reflected Power')
    plt.axhline(y=baseline_power, color='r', linestyle='--', label=f'Baseline ({baseline_power:.2f} dBm)')
    plt.xlabel('gmin value')
    plt.ylabel('Max Reflected Power (dBm)')
    plt.title(f'Case {case_number}: Max Reflected Power vs. gmin', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f'./Figures/max_reflected_power_vs_gmin_po_only_case_{case_number}.png')
    plt.show()

    # Plot for Array Gain
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(max_array_gains_per_segment)))
    
    for i, (segment_id, gains) in enumerate(max_array_gains_per_segment.items()):
        plt.semilogx(gmin_values, gains, marker='s', linestyle='-', color=colors[i], label=f'Optimized Gain (Segment {segment_id})')
    
    for i, (segment_id, gain) in enumerate(baseline_gain_per_segment.items()):
        plt.axhline(y=gain, color=colors[i], linestyle='--', label=f'Baseline Gain (Segment {segment_id}): {gain:.2f} dB')

    plt.xlabel('gmin value')
    plt.ylabel('Max Array Gain (dB)')
    plt.title(f'Case {case_number}: Max Array Gain vs. gmin (Per Segment)', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f'./Figures/max_array_gain_vs_gmin_case_po_only_{case_number}.png')
    plt.show()

def plot_performance_vs_bw(bw_values, max_reflected_powers, max_array_gains_per_segment, baseline_power, baseline_gain_per_segment, case_number):
    """
    Plots the maximum reflected power and array gain as a function of the signal bandwidth for a specific case.
    """
    # Plot for Reflected Power
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(bw_values) / 1e6, max_reflected_powers, marker='o', linestyle='-', label='Optimized Reflected Power')
    plt.axhline(y=baseline_power, color='r', linestyle='--', label=f'Baseline ({baseline_power:.2f} dBm)')
    plt.xlabel('Bandwidth (MHz)')
    plt.ylabel('Max Reflected Power (dBm)')
    plt.title(f'Case {case_number}: Max Reflected Power vs. Bandwidth', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f'./Figures/max_reflected_power_vs_bw_case_{case_number}.png')
    plt.show()

    # Plot for Array Gain
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(max_array_gains_per_segment)))
    
    for i, (segment_id, gains) in enumerate(max_array_gains_per_segment.items()):
        plt.plot(np.array(bw_values) / 1e6, gains, marker='s', linestyle='-', color=colors[i], label=f'Optimized Gain (Segment {segment_id})')
    
    for i, (segment_id, gain) in enumerate(baseline_gain_per_segment.items()):
        plt.axhline(y=gain, color=colors[i], linestyle='--', label=f'Baseline Gain (Segment {segment_id}): {gain:.2f} dB')

    plt.xlabel('Bandwidth (MHz)')
    plt.ylabel('Max Array Gain (dB)')
    plt.title(f'Case {case_number}: Max Array Gain vs. Bandwidth (Per Segment)', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f'./Figures/max_array_gain_vs_bw_case_{case_number}.png')
    plt.show()

# --- Helper Function to Run Phase-Only Solvers ---

def _run_po_solver(solver_obj, po_solver_name, s_data_cube=None, h5_filepath=None, dataset_name=None, device='cpu', max_iter=7000,
                   learning_rate=0.1, momentum=0.9, method_type=None, precomputed_alpha=None, random_init=False):
    """
    A unified function to run either FASTA or iMM solver in in-memory or HDF5 streaming mode.
    Returns: w_opt, record_value (obj_hist), dual_value, primal_value, mu_opt_value
    """
    w_opt = None
    record_value = None
    dual_value = None
    primal_value = None
    mu_opt_value = None
    
    # Validation: Ensure mutually exclusive cube/filepath parameters
    if s_data_cube is not None and h5_filepath is not None:
        raise ValueError("Cannot provide both s_data_cube (in-memory) and h5_filepath (streaming) to solver.")
    
    if po_solver_name == "FASTA":
        # Pass tuning parameters to FASTA
        fasta_method = method_type if method_type else 'PGD' # Default if not specified
        
        if h5_filepath is not None:
            print(f"Using FASTA solver (HDF5 streaming mode) on {h5_filepath}...")
            w_opt, record_value, dual_value, primal_value, mu_opt_value = solver_obj.solver_FASTA(
                h5_filepath=h5_filepath, dataset_name=dataset_name,
                max_iter=max_iter, debug=True, device=device, method_type=fasta_method,
                initial_tau=learning_rate, momentum=momentum
            )
        elif s_data_cube is not None:
            print("Using FASTA solver (in-memory mode)...")
            w_opt, record_value, dual_value, primal_value, mu_opt_value = solver_obj.solver_FASTA(
                s_data_cube=s_data_cube,
                max_iter=max_iter, debug=True, device=device, method_type=fasta_method,
                initial_tau=learning_rate, momentum=momentum
            )
        else:
            print("Error: FASTA requires either an S-data cube or an HDF5 path.")

    elif po_solver_name == "iMM":
        # iMM (solverArraySafety_phaseOnly) only supports in-memory cube
        if h5_filepath is not None:
            print("Error: iMM solver (solverArraySafety_phaseOnly) does not support HDF5 streaming. Requires in-memory cube.")
        elif s_data_cube is not None:
            # Generate random unit-modulus w_init if requested
            w_init = None
            if random_init:
                nEl = solver_obj.nEl
                random_phases = torch.rand(nEl) * 2 * np.pi
                w_init = torch.exp(1j * random_phases).to(torch.complex64)
                print("Using random phase initialization for iMM.")

            print("Using iMM solver (solverArraySafety_phaseOnly) (in-memory mode)...")
            w_opt, record_value, dual_value, primal_value, mu_opt_value = solver_obj.solverArraySafety_phaseOnly(
                S_data_cube=s_data_cube,
                max_iter=max_iter, debug = True, device=device,
                precomputed_alpha=precomputed_alpha,
                w_init=w_init,
            )
        else:
            print("Error: iMM requires an S-data cube in memory.")
    
    else:
        print(f"Warning: Unknown solver '{po_solver_name}'. Skipping optimization.")

    return w_opt, record_value, dual_value, primal_value, mu_opt_value

# --- Main Simulation Logic ---

def run_simulation_case(case_number, gmin=None, po_flag=False, po_solver_name="FASTA", baseline_only=False, random_init=False):
    
    po_str = f", solver={po_solver_name}" if po_flag else ""
    bl_str = " [BASELINE ONLY]" if baseline_only else ""
    print(f"\n----- Running Simulation for Case {case_number} (gmin={gmin}{po_str}){bl_str} -----")
    
    # --- Basic Setup ---
    device = utils.get_default_device()
    print(f"Using device: {device}")
    array_name = "Vivaldi36"
    data_iq = data.Waveform()
    
    tasks = []
    
    # --- Case-Specific Setup ---
    if case_number == 1:
        n_segments = 1
        tasks = [([65], [45])]
        sub_sample = 500
    
    elif case_number == 2:
        n_segments = 2
        tasks = [([65, 45], [145, 45])]
        sub_sample = 500

    elif case_number == 3:
        n_segments = 1
        sub_sample = 500
        # --- Fibonacci Lattice Sampling ---
        # Strictly inside the unit circle
        num_points = 500
        indices = np.arange(0, num_points, dtype=float) + 0.5
        r = np.sqrt(indices / num_points)
        theta_fib = np.pi * (1 + 5**0.5) * indices
        u_samples = r * np.cos(theta_fib)
        v_samples = r * np.sin(theta_fib)

        for u, v in zip(u_samples, v_samples):
            if u**2 + v**2 > 1.0: continue
            theta_rad = np.arcsin(np.sqrt(u**2 + v**2))
            phi_rad = np.arctan2(v, u)
            tasks.append(([np.rad2deg(theta_rad)], [np.rad2deg(phi_rad)]))
        print(f"Generated {len(tasks)} tasks for UV space sweep (Case 3).")
            
    elif case_number == 4:
        n_segments = 4
        sub_sample = 500
        # Angles to create a "cross" pattern with 4 quadrant segments
        # IMPORTANT: These must match the angles used in COMPARE_RIEMANNIAN_METHODS
        thetas = [65, 65, 65, 65]
        phis = [45, 135, -45, -135]
        tasks = [(thetas, phis)]

    elif case_number == 5:
        # --- CASE 5: Two Segments (Seg 1 sweeps UV, Seg 2 fixed) ---
        n_segments = 2
        sub_sample = 500
        
        # Fixed beam for Segment 2
        fixed_theta = 45
        fixed_phi = 45 

        # --- Fibonacci Lattice Sampling (Same as Case 3) ---
        num_points = 5000
        indices = np.arange(0, num_points, dtype=float) + 0.5
        r = np.sqrt(indices / num_points)
        theta_fib = np.pi * (1 + 5**0.5) * indices
        u_samples = r * np.cos(theta_fib)
        v_samples = r * np.sin(theta_fib)

        for u, v in zip(u_samples, v_samples):
            if u**2 + v**2 > 1.0: continue
            theta_rad = np.arcsin(np.sqrt(u**2 + v**2))
            phi_rad = np.arctan2(v, u)
            
            # Task format: ([theta_seg1, theta_seg2], [phi_seg1, phi_seg2])
            current_theta = np.rad2deg(theta_rad)
            current_phi = np.rad2deg(phi_rad)
            
            tasks.append(([current_theta, fixed_theta], [current_phi, fixed_phi]))
            
        print(f"Generated {len(tasks)} tasks for Case 5 (Seg 1 sweeping, Seg 2 fixed at {fixed_theta},{fixed_phi}).")

    else:
        print(f"Case {case_number} is not defined.")
        return None, None, None, None

    # --- Initialize Antenna and Data Storage ---
    antenna = antennaeArray.Array(arrayName=array_name, num_segments=n_segments)
    idx_of_S_freq_in_data_freq = antenna.get_coupling_freq_idx(data_iq.freq_signal)
    S_fc = antenna.get_center_freq_coupling_matrix(data_iq.fc)
    
    all_results_opt_amp = []
    broadside_weights = np.ones(shape=(antenna.nEl,))
    print(f"Starting {len(tasks)} task(s)...")

    # --- Pre-load or compute S_data_cube once (shared across all tasks and cases) ---
    # The cube depends only on the antenna S-parameters and waveform — not on
    # look direction, angles, or number of segments — so one file suffices for all cases.
    s_data_cube_tensor_cpu = None
    use_hdf5_streaming = False
    s_data_cube_filename = f"S_data_cube_{array_name}.h5"
    s_data_cube_dataset = "s_cube"
    max_iter = 3000

    if po_flag and not baseline_only:
        MEMORY_THRESHOLD_RATIO = 0.95
        available_ram = psutil.virtual_memory().available
        file_exists = os.path.exists(s_data_cube_filename)
        file_is_valid = False
        cube_bytes = 0

        if file_exists:
            try:
                with h5py.File(s_data_cube_filename, 'r') as hf:
                    dataset = hf[s_data_cube_dataset]
                    cube_bytes = np.prod(dataset.shape) * np.dtype(dataset.dtype).itemsize
                file_is_valid = True
                if cube_bytes > available_ram * MEMORY_THRESHOLD_RATIO:
                    if po_solver_name == "iMM":
                        print("Error: iMM solver requires in-memory cube. Cube size is too large.")
                        return None, None, None, None
                    use_hdf5_streaming = True
                else:
                    with h5py.File(s_data_cube_filename, 'r') as hf:
                        s_data_cube_numpy = hf[s_data_cube_dataset][:]
                    s_data_cube_tensor_cpu = torch.from_numpy(s_data_cube_numpy)
                    del s_data_cube_numpy
            except Exception:
                try: os.remove(s_data_cube_filename)
                except: pass
                file_exists = False
                file_is_valid = False

        if not file_exists or not file_is_valid:
            print(f"Computing S_data_cube and saving to {s_data_cube_filename}...")
            # array_man_freq is unused inside calculate_reflected_power_directly when
            # building the cube, so any valid manifold works; we use broadside (theta=phi=0).
            _dummy_manifold = antenna.calculate_array_manifold(
                data_iq.freq_signal, get_angle=True,
                theta_deg=[0] * n_segments, phi_deg=[0] * n_segments
            )
            _, _, s_data_cube_tensor_cpu = antenna.calculate_reflected_power_directly(
                data_iq, _dummy_manifold, broadside_weights, return_cube=True,
                batch_size=512, device="cpu", verbose=False
            )
            cube_bytes = s_data_cube_tensor_cpu.nelement() * s_data_cube_tensor_cpu.element_size()
            s_data_cube_numpy = s_data_cube_tensor_cpu.numpy()
            with h5py.File(s_data_cube_filename, 'w') as hf:
                hf.create_dataset(
                    name=s_data_cube_dataset, shape=s_data_cube_numpy.shape,
                    dtype=s_data_cube_numpy.dtype,
                    chunks=(256, s_data_cube_numpy.shape[1], s_data_cube_numpy.shape[2]),
                    compression='gzip', data=s_data_cube_numpy
                )
            del s_data_cube_numpy
            if cube_bytes > available_ram * MEMORY_THRESHOLD_RATIO:
                if po_solver_name == "iMM":
                    print("Error: iMM solver requires in-memory cube. Cube size is too large.")
                    return None, None, None, None
                del s_data_cube_tensor_cpu
                s_data_cube_tensor_cpu = None
                use_hdf5_streaming = True

    # Precompute and cache alpha (max eigenvalues) for iMM solver
    precomputed_alpha = None
    if po_flag and not baseline_only and po_solver_name == "iMM" and s_data_cube_tensor_cpu is not None:
        from run_speed_comparison import precompute_and_cache_alpha
        precomputed_alpha = precompute_and_cache_alpha(s_data_cube_tensor_cpu, s_data_cube_filename)

    task_iterator = tqdm(tasks, desc=f"Case {case_number} Progress") if len(tasks) > 1 else tasks
    for theta_list, phi_list in task_iterator:
        
        # Calculate these common parameters first
        look_direction_freq = antenna.calculate_array_manifold(
            data_iq.freq_signal, get_angle=True, theta_deg=theta_list, phi_deg=phi_list
        )
        
        # --- Pre-calculate UV coordinates (For plotting heatmaps later) ---
        u, v = (None, None)
        if case_number == 3:
            # For Case 3, there's only 1 angle
            theta_rad, phi_rad = np.deg2rad(theta_list[0]), np.deg2rad(phi_list[0])
            u, v = np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad)
        elif case_number == 5:
            # For Case 5, we save the UV of the *sweeping* segment (Segment 0)
            theta_rad, phi_rad = np.deg2rad(theta_list[0]), np.deg2rad(phi_list[0])
            u, v = np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad)
        
        # --- BASELINE ONLY PATH ---
        if baseline_only:
            # Use the center frequency array manifold as baseline weights
            # so the beam is steered to the current look direction
            a_c_baseline = antenna.calculate_array_manifold(
                data_iq.fc, get_angle=True, theta_deg=theta_list, phi_deg=phi_list
            )
            baseline_w = a_c_baseline * broadside_weights

            # Calculate Baseline Metrics directly
            # Returns: (reflected_coeff_per_port_squared, S_cube)
            refl_coeff_sq, _ = antenna.calculate_reflected_power_directly(
                data_iq, look_direction_freq, baseline_w, batch_size=128, device=device, verbose=False
            )
            # Convert to numpy for storage
            refl_coeff_sq_np = refl_coeff_sq.cpu().numpy() if isinstance(refl_coeff_sq, torch.Tensor) else refl_coeff_sq

            max_ref_pow = 30 + 10 * np.log10(np.max(refl_coeff_sq_np))
            max_array_ref_gain_dict = antenna.plotArrayResponse(
                baseline_w, theta_list, phi_list, data_iq.fc, n_gridpoints_per_u_v=100, verbose=False, plot=False
            )

            all_results_opt_amp.append({
                'case': case_number,
                'u': u, 'v': v,
                'w_opt': baseline_w,
                'max_reflected_power_dbm': max_ref_pow,
                'max_array_gain_db': max_array_ref_gain_dict,
                'max_ref_reflected_power_dbm': max_ref_pow,
                'max_ref_array_gain_db': max_array_ref_gain_dict,
                'reflection_coeff_sq_per_port': refl_coeff_sq_np
            })
            continue

        # --- OPTIMIZATION PATH ---
        a_c = antenna.calculate_array_manifold(data_iq.fc, get_angle=True, theta_deg=theta_list, phi_deg=phi_list)
        solver_obj = solver.SolveArraySafety(S_fc, a_c, verbose=False, gmin=gmin)
        
        w_opt = None 
        record_value = None 
        dual_value = None
        primal_value = None
        mu_opt_value = None
        
        if po_flag:
            # Cube was pre-loaded once before the task loop — just call the solver.
            if use_hdf5_streaming:
                w_opt, record_value, dual_value, primal_value, mu_opt_value = _run_po_solver(
                    solver_obj, po_solver_name, h5_filepath=s_data_cube_filename,
                    dataset_name=s_data_cube_dataset, device=device, max_iter=max_iter,
                    precomputed_alpha=precomputed_alpha, random_init=random_init,
                )
            elif s_data_cube_tensor_cpu is not None:
                w_opt, record_value, dual_value, primal_value, mu_opt_value = _run_po_solver(
                    solver_obj, po_solver_name, s_data_cube=s_data_cube_tensor_cpu,
                    device=device, max_iter=max_iter,
                    precomputed_alpha=precomputed_alpha, random_init=random_init,
                )
        
        else:
            # INF NORM (Non-Phase Only) — Gram matrix reformulation uses all frequencies
            w_opt, _ = solver_obj.solveArraySafety_inf(
                antenna.Sf, idx_of_S_freq_in_data_freq,
                data_iq=data_iq
            )

        if w_opt is None:
            print("Warning: Optimization failed. Skipping task.")
            return None, None, None, None

        # --- Post-Optimization Processing ---
        if w_opt.ndim == 1:
            if isinstance(w_opt, torch.Tensor): w_opt = w_opt.cpu().numpy()
            w_opt = np.expand_dims(w_opt,axis=1)

        # Baseline Calculation (for comparison)
        # Returns: (reflected_coeff_per_port_squared, S_cube)
        refl_coeff_sq_base, _ = antenna.calculate_reflected_power_directly(data_iq, look_direction_freq, broadside_weights, batch_size=128, device=device, verbose=False)
        refl_coeff_sq_base_np = refl_coeff_sq_base.cpu().numpy() if isinstance(refl_coeff_sq_base, torch.Tensor) else refl_coeff_sq_base

        max_ref_pow = 30 + 10 * np.log10(np.max(refl_coeff_sq_base_np))
        max_array_ref_gain_dict = antenna.plotArrayResponse(broadside_weights, theta_list, phi_list, data_iq.fc, n_gridpoints_per_u_v=100, verbose=False, plot=False)

        # Optimized Calculation
        w_opt_flat = w_opt[:, 0]
        if not isinstance(w_opt_flat, torch.Tensor):
             w_opt_flat = w_opt_flat.astype(np.complex128)

        refl_coeff_sq_opt, _ = antenna.calculate_reflected_power_directly(data_iq, look_direction_freq, w_opt_flat, batch_size=128, device=device, verbose=False)
        refl_coeff_sq_opt_np = refl_coeff_sq_opt.cpu().numpy() if isinstance(refl_coeff_sq_opt, torch.Tensor) else refl_coeff_sq_opt

        max_reflected_power_dbm = 30 + 10 * np.log10(np.max(refl_coeff_sq_opt_np))
        max_array_gain_db_dict = antenna.plotArrayResponse(w_opt_flat, theta_list, phi_list, data_iq.fc, n_gridpoints_per_u_v=100, verbose=False, plot=False)
        
        if po_flag and po_solver_name == "iMM" and record_value is not None:
             full_record = {'obj_history': record_value, 'dual_history': dual_value, 'primal_history': primal_value, 'mu_history': mu_opt_value}
             scipy.io.savemat(f"convergence_results_{po_solver_name}_a_c_init_case_{case_number}.mat", full_record)
        elif po_flag and po_solver_name == "FASTA" and record_value is not None:
             scipy.io.savemat(f"convergence_results_{po_solver_name}_case_{case_number}.mat", {'obj_history': record_value})

        # Append Results
        all_results_opt_amp.append({
            'case': case_number, 'u': u, 'v': v,
            'w_opt': w_opt[:, 0],
            'max_reflected_power_dbm': max_reflected_power_dbm,
            'max_array_gain_db': max_array_gain_db_dict,
            'max_ref_reflected_power_dbm': max_ref_pow,
            'max_ref_array_gain_db': max_array_ref_gain_dict,
            'reflection_coeff_sq_per_port_opt': refl_coeff_sq_opt_np,
            'reflection_coeff_sq_per_port_base': refl_coeff_sq_base_np
        })
        
    # Release the shared cube now that all tasks are done
    if s_data_cube_tensor_cpu is not None:
        del s_data_cube_tensor_cpu
    gc.collect()

    # --- File Saving ---
    results_dir = "./Weights/Gmin/inf_norm_w" if not po_flag else "./Weights/Gmin/phase_only_w"
    os.makedirs(results_dir, exist_ok=True)

    if baseline_only:
        filename_base = f"baseline_results_case_{case_number}"
    else:
        gmin_str = f"_gmin_{gmin:.2e}" if gmin is not None else ""
        po_solver_str = f"_{po_solver_name}" if po_flag else ""
        filename_base = f"optimized_results{po_solver_str}_case_{case_number}{gmin_str}"

    # Save as pickle
    pkl_filename = os.path.join(results_dir, filename_base + ".pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(all_results_opt_amp, f)

    # Save as mat file (convert to suitable format for MATLAB)
    mat_filename = os.path.join(results_dir, filename_base + ".mat")
    mat_data = {}
    if all_results_opt_amp:
        # Extract data from results
        for key in all_results_opt_amp[0].keys():
            if key in ['case', 'u', 'v']:
                # Scalar values - create array, replacing None with nan for MATLAB compatibility
                values = [result.get(key) for result in all_results_opt_amp]
                mat_data[key] = np.array([v if v is not None else np.nan for v in values], dtype=float)
            elif key in ['reflection_coeff_sq_per_port_opt', 'reflection_coeff_sq_per_port_base',
                        'reflection_coeff_sq_per_port']:
                # Per-port vectors - stack them
                per_port_data = np.array([result.get(key) for result in all_results_opt_amp])
                mat_data[key] = per_port_data
            elif key == 'w_opt':
                # Weight vectors
                w_opt_data = np.array([result.get(key) for result in all_results_opt_amp])
                mat_data[key] = w_opt_data
            elif isinstance(all_results_opt_amp[0].get(key), dict):
                # Skip dictionary types (max_array_gain_db, etc.) as they don't serialize well to mat
                pass
            else:
                try:
                    # Scalar values
                    mat_data[key] = np.array([result.get(key) for result in all_results_opt_amp])
                except:
                    pass

    scipy.io.savemat(mat_filename, mat_data)
    print(f"Results saved to:\n  Pickle: {pkl_filename}\n  MAT: {mat_filename}")
    
    if all_results_opt_amp:
        first_opt_result = all_results_opt_amp[0]
        if isinstance(first_opt_result['max_ref_reflected_power_dbm'], (np.ndarray, torch.Tensor)):
             base_power_val = np.max(first_opt_result['max_ref_reflected_power_dbm']) if isinstance(first_opt_result['max_ref_reflected_power_dbm'], np.ndarray) else first_opt_result['max_ref_reflected_power_dbm'].cpu().numpy().max()
             base_power_val = 30 + 10 * np.log10(base_power_val)
        else:
             base_power_val = first_opt_result['max_ref_reflected_power_dbm']

        return (
            first_opt_result['max_reflected_power_dbm'],
            first_opt_result['max_array_gain_db'], 
            base_power_val,
            first_opt_result['max_ref_array_gain_db']
        )
    else:
        return None, None, None, None

def main():
    """
    Main function to run the antenna array analysis.
    """
    
    # --- Control Flags ---
    BASELINE_ONLY = False  # Set to True to calculate ONLY baseline performance (no optimization)
    COMPARE_RIEMANNIAN_METHODS = False
    PERFORM_GMIN_SWEEP = False
    PERFORM_BW_SWEEP = False

    po_only = True
    PHASE_ONLY_SOLVER = "iMM" # Options: "FASTA", "iMM"
    RANDOM_INIT = True  # Set to True to use random phase initialization instead of a_c for iMM

    # Choose which case(s) to run for sweep/single run
    CASES_TO_RUN = [4]
    
    GMIN_VALUES_TO_SWEEP = [1296/3,1296/2,1296/1.5,1296/1.3,1296/1.2,1296/1.1,1296/1] 
    BW_VALUES_TO_SWEEP = np.linspace(100e6, 700e6, 10)

    # --- Directory Setup ---
    base_fig_dir = "./Figures/Gmin/phase_only_w" if po_only else "./Figures/Gmin/inf_norm_w"
    base_weights_dir = "./Weights/Gmin/phase_only_w" if po_only else "./Weights/Gmin/inf_norm_w"
    os.makedirs(base_fig_dir, exist_ok=True)
    os.makedirs(base_weights_dir, exist_ok=True)

    if COMPARE_RIEMANNIAN_METHODS:
        print("🚀 Starting Riemannian Method Comparison mode.")
        
        # --- TUNING HYPERPARAMETERS ---
        LEARNING_RATE_TO_TUNE = 100 # initial_tau
        MOMENTUM_TO_TUNE = 0.1      # beta
        INCLUDE_IMM_IN_COMPARISON = True # New flag to include iMM
        # -----------------------------

        # Config for the comparison
        case_for_comparison = 4
        base_iterations = 3000
        gmin_val = None 
        
        methods_to_compare = ['PGD_momentum']
        
        convergence_results = {}
        weights_results = {} 
        
        print(f"--- Setting up for Case {case_for_comparison} (LR={LEARNING_RATE_TO_TUNE}, Mom={MOMENTUM_TO_TUNE}) ---")
        
        # --- 1. Setup ---
        device = utils.get_default_device()
        array_name = "Vivaldi36"
        data_iq = data.Waveform()
        
        # Consistent Case 4 Definition
        if case_for_comparison == 4:
            n_segments = 4
            thetas = [65, 65, 65, 65]
            phis = [45, 135, -45, -135]
            tasks = [(thetas, phis)]
        elif case_for_comparison == 2:
            n_segments = 2
            tasks = [([45, 45], [45, 135])]
        elif case_for_comparison == 1:
            n_segments = 1
            tasks = [([65], [45])]
        else:
            print(f"Warning: Case {case_for_comparison} default setup.")
            n_segments = 2
            tasks = [([45, 45], [45, 135])]

        antenna = antennaeArray.Array(arrayName=array_name, num_segments=n_segments)
        S_fc = antenna.get_center_freq_coupling_matrix(data_iq.fc)
        
        theta, phi = tasks[0]
        look_direction_freq = antenna.calculate_array_manifold(data_iq.freq_signal, get_angle=True, theta_deg=theta, phi_deg=phi)
        a_c = antenna.calculate_array_manifold(data_iq.fc, get_angle=True, theta_deg=theta, phi_deg=phi)
        broadside_weights = np.ones(shape=(antenna.nEl,))

        solver_obj = solver.SolveArraySafety(S_fc, a_c, verbose=False, gmin=gmin_val)

        # --- 2. Load or Generate S_data_cube ---
        s_data_cube_tensor_cpu = None
        
        # --- MODIFICATION: Append case number to filename ---
        s_data_cube_filename = f"S_data_cube_{array_name}.h5"
        s_data_cube_dataset = "s_cube"
        MEMORY_THRESHOLD_RATIO = 0.95 
        file_exists = os.path.exists(s_data_cube_filename)
        use_hdf5 = False

        available_ram = psutil.virtual_memory().available

        if file_exists:
            print(f"Data cube file found: {s_data_cube_filename}. Checking size...")
            try:
                with h5py.File(s_data_cube_filename, 'r') as hf:
                    cube_bytes = np.prod(hf[s_data_cube_dataset].shape) * np.dtype(hf[s_data_cube_dataset].dtype).itemsize
                
                print(f"Data cube size on disk: {cube_bytes / 1e9:.2f} GB | Available RAM: {available_ram / 1e9:.2f} GB")

                if cube_bytes > available_ram * MEMORY_THRESHOLD_RATIO:
                    print("Data cube is too large to load into RAM. Using HDF5 on-disk mode.")
                    use_hdf5 = True
                else:
                    print("Data cube fits in RAM. Loading from file...")
                    with h5py.File(s_data_cube_filename, 'r') as hf:
                        s_data_cube_tensor_cpu = torch.from_numpy(hf[s_data_cube_dataset][:])
                    print("Load complete.")
            
            except Exception as e:
                print(f"Error reading HDF5 file: {e}. Deleting and recalculating.")
                if os.path.exists(s_data_cube_filename): os.remove(s_data_cube_filename)
                file_exists = False
        
        if not file_exists:
            print("Data cube file not found. Calculating and saving...")
            _, _, s_data_cube_tensor_cpu = antenna.calculate_reflected_power_directly(
                data_iq, look_direction_freq, broadside_weights, return_cube=True,
                batch_size=512, device="cpu", verbose=False
            )
            print("Saving new data cube...")
            s_data_cube_numpy = s_data_cube_tensor_cpu.numpy()
            with h5py.File(s_data_cube_filename, 'w') as hf:
                hf.create_dataset(
                    s_data_cube_dataset, data=s_data_cube_numpy, compression='gzip',
                    chunks=(256, s_data_cube_numpy.shape[1], s_data_cube_numpy.shape[2])
                )
            del s_data_cube_numpy
            print("Save complete.")

            cube_bytes = s_data_cube_tensor_cpu.nelement() * s_data_cube_tensor_cpu.element_size()
            if cube_bytes > available_ram * MEMORY_THRESHOLD_RATIO:
                 print("New cube is too large for RAM. Deleting tensor and using HDF5 streaming.")
                 del s_data_cube_tensor_cpu
                 s_data_cube_tensor_cpu = None
                 use_hdf5 = True
            
        # --- 3. Run Comparison ---

        # Precompute and cache alpha for iMM
        precomputed_alpha_cmp = None
        if s_data_cube_tensor_cpu is not None:
            from run_speed_comparison import precompute_and_cache_alpha
            precomputed_alpha_cmp = precompute_and_cache_alpha(s_data_cube_tensor_cpu, s_data_cube_filename)

        # 3a. Run iMM if requested
        if INCLUDE_IMM_IN_COMPARISON:
            print(f"\n--- Running Method: iMM (Max Iter: {base_iterations}) ---")

            # iMM requires in-memory cube. If we are in HDF5 mode, this will fail or skip.
            if s_data_cube_tensor_cpu is not None:
                w_opt_imm, obj_hist_imm, _, _, _ = _run_po_solver(
                    solver_obj, "iMM", s_data_cube=s_data_cube_tensor_cpu,
                    device=device, max_iter=base_iterations,
                    precomputed_alpha=precomputed_alpha_cmp,
                )
                if obj_hist_imm is not None:
                    convergence_results["iMM"] = obj_hist_imm
                    if isinstance(w_opt_imm, torch.Tensor):
                        w_opt_imm = w_opt_imm.cpu().numpy()
                    weights_results["iMM"] = w_opt_imm
                    print(f"Final objective value for iMM: {obj_hist_imm[-1]:.4f} dBm")
            else:
                print("Warning: iMM skipped because S_data_cube is not in memory (HDF5 mode).")

        # 3b. Run Gradient Methods
        for method in methods_to_compare:
            # Adjust iteration count: 2x for no_momentum
            current_max_iter = base_iterations * 2 if 'no_momentum' in method else base_iterations
            
            print(f"\n--- Running Method: {method} (Max Iter: {current_max_iter}) ---")
            
            # --- Logic Mapping ---
            # Default logic for RGD methods
            solver_method_type = method 
            solver_momentum = 0.0
            
            if method == 'PGD_momentum':
                solver_method_type = 'PGD'
                solver_momentum = MOMENTUM_TO_TUNE
            elif method == 'PGD':
                solver_method_type = 'PGD'
                solver_momentum = 0.0
            elif method == 'RGD_momentum':
                solver_method_type = 'RGD_momentum'
                solver_momentum = MOMENTUM_TO_TUNE # Assuming RGD also uses this beta if implemented
            elif method == 'RGD_no_momentum':
                solver_method_type = 'RGD_no_momentum'
                solver_momentum = 0.0

            # Pass learning_rate and momentum to helper
            if use_hdf5:
                w_opt, obj_history, _, _, _ = _run_po_solver(
                    solver_obj, "FASTA", h5_filepath=s_data_cube_filename, 
                    dataset_name=s_data_cube_dataset, device=device, max_iter=current_max_iter,
                    learning_rate=LEARNING_RATE_TO_TUNE, momentum=solver_momentum,
                    method_type=solver_method_type
                )
            else:
                if s_data_cube_tensor_cpu is None:
                    print(f"Error: In-memory mode failed for {method}. Cube was too large and tensor was deleted. Skipping.")
                    continue

                w_opt, obj_history, _, _, _ = _run_po_solver(
                    solver_obj, "FASTA", s_data_cube=s_data_cube_tensor_cpu,
                    device=device, max_iter=current_max_iter,
                    learning_rate=LEARNING_RATE_TO_TUNE, momentum=solver_momentum,
                    method_type=solver_method_type
                )
            
            convergence_results[method] = obj_history
            print(f"Final objective value for {method}: {obj_history[-1]:.4f} dBm")
            
            # --- Save Weights Logic ---
            if isinstance(w_opt, torch.Tensor):
                w_opt_np = w_opt.cpu().numpy()
            else:
                w_opt_np = w_opt
            weights_results[method] = w_opt_np
        
        # --- Construct Filenames with Hyperparameters ---
        param_str = f"lr_{LEARNING_RATE_TO_TUNE}_mom_{MOMENTUM_TO_TUNE}"
        mat_filename = f"convergence_results_{param_str}_case_{case_for_comparison}.mat"
        pkl_filename = f"weights_comparison_{param_str}_case_{case_for_comparison}.pkl"
        plot_filename = os.path.join(base_fig_dir, f'convergence_{param_str}_case_{case_for_comparison}.png')

        # Save Convergence Data (MAT)
        scipy.io.savemat(mat_filename, convergence_results)
        
        # Save Weights Data (Pickle and MAT)
        # 1. Pickle (for Python loading)
        with open(pkl_filename, "wb") as f:
            pickle.dump(weights_results, f)
        
        # 2. Add to the MAT file (for MATLAB loading)
        combined_results = convergence_results.copy()
        for k, v in weights_results.items():
            combined_results[f"w_opt_{k}"] = v
        scipy.io.savemat(mat_filename, combined_results)
        print(f"Results saved to {mat_filename} and {pkl_filename}")
        
        # --- 4. Plot Results ---
        plt.figure(figsize=(12, 8))
        for method, history in convergence_results.items():
            plt.plot(history, label=method, alpha=0.9)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Objective Value (dBm)', fontsize=12)
        plt.title(f'Convergence (Case {case_for_comparison})\nLR={LEARNING_RATE_TO_TUNE}, Mom={MOMENTUM_TO_TUNE}', fontsize=14)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        if convergence_results:
             min_start_val = min(h[int(base_iterations*0.1)] for h in convergence_results.values() if len(h) > int(base_iterations*0.1))
             plt.ylim(bottom=min_start_val - 5) 
        
        plt.savefig(plot_filename)
        plt.show()

        if s_data_cube_tensor_cpu is not None:
            del s_data_cube_tensor_cpu
        gc.collect()

    elif PERFORM_GMIN_SWEEP and not BASELINE_ONLY:
        print("🚀 Starting GMIN SWEEP mode.")
        sweep_results = {} 
        
        for case in CASES_TO_RUN:
            print(f"\n--- Sweeping gmin for Case {case} ---")
            sweep_results[case] = {
                'gmin': [], 'power': [], 'gain_per_segment': {}, 
                'baseline_power': None, 'baseline_gain_per_segment': {}
            }

            for gmin_val in tqdm(GMIN_VALUES_TO_SWEEP, desc=f"gmin sweep Case {case}"):
                opt_power, opt_gain_dict, base_power, base_gain_dict = run_simulation_case(
                    case,
                    gmin=gmin_val,
                    po_flag=po_only,
                    po_solver_name=PHASE_ONLY_SOLVER,
                    random_init=RANDOM_INIT
                )
                
                if opt_power is not None:
                    sweep_results[case]['gmin'].append(gmin_val)
                    sweep_results[case]['power'].append(opt_power)
                    
                    for segment_id, gain in opt_gain_dict.items():
                        if segment_id not in sweep_results[case]['gain_per_segment']:
                            sweep_results[case]['gain_per_segment'][segment_id] = []
                        sweep_results[case]['gain_per_segment'][segment_id].append(gain)

                    if sweep_results[case]['baseline_power'] is None:
                        sweep_results[case]['baseline_power'] = base_power
                        sweep_results[case]['baseline_gain_per_segment'] = base_gain_dict
            
            if sweep_results[case]['gmin']:
                plot_performance_vs_gmin(
                    gmin_values=sweep_results[case]['gmin'],
                    max_reflected_powers=sweep_results[case]['power'],
                    max_array_gains_per_segment=sweep_results[case]['gain_per_segment'],
                    baseline_power=sweep_results[case]['baseline_power'],
                    baseline_gain_per_segment=sweep_results[case]['baseline_gain_per_segment'],
                    case_number=case
                )

    elif PERFORM_BW_SWEEP and not BASELINE_ONLY:
        print("🚀 Starting BANDWIDTH SWEEP mode.")
        sweep_results = {} 

        N_FREQ_OPT = 70

        for case in CASES_TO_RUN:
            print(f"\n--- Sweeping bandwidth for Case {case} ---")
            sweep_results[case] = {
                'bw': [], 'power': [], 'gain_per_segment': {},
                'baseline_power': None, 'baseline_gain_per_segment': {}
            }

            if case == 1: n_segments, thetas, phis, sub_sample = 1, [65], [45], 500
            elif case == 2: n_segments, thetas, phis, sub_sample = 2, [45, 45], [45, 135], 500
            elif case == 3:
                print("Warning: Bandwidth sweep for Case 3 (UV sweep) is not supported. Skipping.")
                continue
            elif case == 4: n_segments, thetas, phis, sub_sample = 4, [45, 45, 45, 45], [45, 135, -45, -135], 500
            else:
                print(f"Case {case} is not defined for BW sweep. Skipping.")
                continue

            device = utils.get_default_device()
            antenna = antennaeArray.Array(arrayName="Vivaldi36", num_segments=n_segments)

            for bw_val in tqdm(BW_VALUES_TO_SWEEP, desc=f"Bandwidth sweep Case {case}"):
                data_iq = data.Waveform()
                data_iq.generate_waveform(bw_val)

                idx_of_S_freq_in_data_freq = antenna.get_coupling_freq_idx(data_iq.freq_signal)
                S_fc = antenna.get_center_freq_coupling_matrix(data_iq.fc)
                
                look_direction_freq = antenna.calculate_array_manifold(data_iq.freq_signal, get_angle=True, theta_deg=thetas, phi_deg=phis)
                a_c = antenna.calculate_array_manifold(data_iq.fc, get_angle=True, theta_deg=thetas, phi_deg=phis)

                solver_obj = solver.SolveArraySafety(S_fc, a_c, verbose=False)

                if len(idx_of_S_freq_in_data_freq) > N_FREQ_OPT:
                    sub_sample = int(len(idx_of_S_freq_in_data_freq) / N_FREQ_OPT)
                else:
                    sub_sample = 1
                
                w_opt, _ = solver_obj.solveArraySafety_inf(
                    antenna.Sf, idx_of_S_freq_in_data_freq,
                    sub_sample=sub_sample, data_iq=data_iq
                )
                
                if w_opt is None:
                    continue

                if w_opt.ndim == 2 and w_opt.shape[1] == 1:
                    w_opt_flat = w_opt[:, 0]
                elif w_opt.ndim == 1:
                    w_opt_flat = w_opt
                else:
                    continue

                refl_coeff_sq_opt, _ = antenna.calculate_reflected_power_directly(data_iq, look_direction_freq, w_opt_flat, batch_size=128, device=device, verbose=False)
                opt_power = 30 + 10 * np.log10(np.max(refl_coeff_sq_opt.cpu().numpy()))
                opt_gain_dict = antenna.plotArrayResponse(w_opt_flat, thetas, phis, data_iq.fc, n_gridpoints_per_u_v=100, verbose=False, plot=False)

                all_results_opt_amp = [{
                    'case': case,
                    'bandwidth_Hz': bw_val,
                    'w_opt': w_opt_flat,
                    'max_reflected_power_dbm': opt_power,
                    'max_array_gain_db': opt_gain_dict
                }]
                
                bw_str = f"_bw_{bw_val/1e6:.0f}MHz"
                results_dir = "./Weights/Gmin/inf_norm_w"  
                os.makedirs(results_dir, exist_ok=True)
                
                with open(os.path.join(results_dir, f"optimized_results_case_{case}{bw_str}.pkl"), 'wb') as f:
                    pickle.dump(all_results_opt_amp, f)

                sweep_results[case]['bw'].append(bw_val)
                sweep_results[case]['power'].append(opt_power)
                for seg_id, gain in opt_gain_dict.items():
                    if seg_id not in sweep_results[case]['gain_per_segment']:
                        sweep_results[case]['gain_per_segment'][seg_id] = []
                    sweep_results[case]['gain_per_segment'][seg_id].append(gain)

            _, _, base_power, base_gain_dict = run_simulation_case(case) 
            
            if sweep_results[case]['bw']:
                plot_performance_vs_bw(
                    bw_values=sweep_results[case]['bw'], max_reflected_powers=sweep_results[case]['power'],
                    max_array_gains_per_segment=sweep_results[case]['gain_per_segment'], baseline_power=base_power,
                    baseline_gain_per_segment=base_gain_dict, case_number=case
                )

    # --- Single Run or Baseline Run Mode ---
    else:
        mode_str = "BASELINE ONLY" if BASELINE_ONLY else f"SINGLE RUN (Solver: {PHASE_ONLY_SOLVER if po_only else 'inf_norm'})"
        print(f"🚀 Starting {mode_str} mode.")
        
        for case in CASES_TO_RUN:
            run_simulation_case(
                case,
                po_flag=po_only,
                po_solver_name=PHASE_ONLY_SOLVER,
                baseline_only=BASELINE_ONLY,
                random_init=RANDOM_INIT
            )

if __name__ == "__main__":
    main()