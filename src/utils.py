import torch 
import numpy as np
import math 
from tqdm import tqdm
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def create_animation(all_results, antenna, data_iq, output_filename="array_response_sweep.mp4", fps=10, dpi=150):
    """
    Creates and saves an animation of the antenna array response over a sweep of angles.

    Args:
        all_results (list): A list of dictionaries, where each dictionary contains the
                            results ('theta_deg', 'phi_deg', 'w_opt') for one angle.
        antenna (object): The antenna array object, which has the plotArrayResponse method.
        data_iq (object): The waveform data object required by plotArrayResponse.
        output_filename (str): The name of the output movie file.
        fps (int): Frames per second for the output movie.
        dpi (int): Dots per inch for the output movie.
    """
    # Ensure the output directory exists
    output_dir = "./Movies"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)

    print("Creating array response animation...")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    def update(frame):
        """Helper function to update each frame of the animation."""
        ax.clear()
        result = all_results[frame]
        theta_val, phi_val = result['theta_deg'], result['phi_deg']
        w_opt_frame = result['w_opt']

        # Plot the array response for the current frame's weights
        antenna.plotArrayResponse(
            w_opt_frame, [theta_val], [phi_val], data_iq.fc,
            n_gridpoints_per_u_v=100, verbose=False, plot=True, ax=ax
        )
        ax.set_title(f"Array Response @ Theta={theta_val:.1f}°, Phi={phi_val:.1f}°")

    # Create the animation object
    num_frames = len(all_results)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)

    # Save the animation
    try:
        ani.save(full_path, writer='ffmpeg', fps=fps, dpi=dpi,
                 progress_callback=lambda i, n: print(f'Saving frame {i + 1} of {n}'))
        print(f"Animation saved successfully to '{full_path}'")
    except FileNotFoundError:
        print("\nERROR: `ffmpeg` not found.")
        print("Please install ffmpeg to save the animation.")
        print("See: https://ffmpeg.org/download.html")

    plt.close(fig) # Prevent the final plot from displaying

def get_default_device():
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda:0') # Change it to 1,2 and so on if you have multiple GPUs [Run nvidia-smi in terminal to check]
        print("CUDA is available. Using GPU.")
    # Check for Apple's Metal Performance Shaders (MPS) availability
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        #device = "cpu"
        print("MPS is available. Using Apple's Metal. Certain operations may not be supported/correct on MPS, so ensure compatibility.")
    # Default to CPU if neither CUDA nor MPS is available
    else:
        device = torch.device('cpu')
        print("CUDA and MPS are not available. Using CPU.")
    
    return device

def get_nearest_psd_matrix(P):
    """
    Finds the nearest positive semidefinite matrix for a batch of matrices P.

    Performs an eigenvalue decomposition of the nearest Hermitian matrix to P,
    clamps any negative eigenvalues to zero, and reconstructs the matrix.
    This is a standard method to project a matrix onto the PSD cone.

    Args:
        P (array-like): A batch of square matrices, shape (B, N, N).

    Returns:
        np.ndarray: The nearest PSD matrix for each matrix in the batch, shape (B, N, N).
    """
    # 1. Convert to tensor and enforce the matrix is Hermitian to ensure real eigenvalues.
    P_tensor = torch.as_tensor(P, dtype=torch.complex128, device="cpu")
    P_hermitian = (P_tensor + P_tensor.mH) / 2.0
    
    # 2. Perform eigenvalue decomposition. 
    #    torch.linalg.eigh is for Hermitian matrices, handles batches, and is stable.
    eigenvalues, eigenvectors = torch.linalg.eigh(P_hermitian)
    
    # 3. Clamp small negative eigenvalues to zero.
    # Note: eigenvalues from a Hermitian matrix are always real.
    eigenvalues.clamp_(min=0)
    
    # 4. Reconstruct the matrix from the modified eigenvalues and original eigenvectors.
   
    P_psd = torch.einsum('bij,bj,bjk->bik', eigenvectors, eigenvalues,eigenvectors.mH)
   
    return P_psd


def createFrequencyCombinedMatrix(S_band,freq_S_indices,array_man,batch_size = 32,dataIq = None,subsampling = 2, device = "cpu"):
    torch.set_grad_enabled(False)
    """
    Returns a 3D coupling matrix scaled by the frequency magnitude of the waveform and summed over all frequencies.
    The array manifold drift over frequencies is applied to the S-band data.
    Output shape is (n_ports, n_ports, n_ports). Where the last two dimensions tell how all the ports impact the port number given in the 
    first dimension.
    """
    S_band = torch.asarray(S_band, dtype=torch.complex128,device= device)
    array_man = np.squeeze(array_man)
    array_man = array_man.T
    array_man = torch.asarray(array_man, dtype=torch.complex128,device= device)
    if dataIq is not None:
        power_iq = torch.asarray(dataIq.get_fft_signal() ** 2,dtype = torch.complex128,device = device)

    
    n_ports = S_band.shape[0]
    sum_s = torch.zeros((n_ports, n_ports, n_ports), dtype=torch.complex128, device = device)

    freq_S_indices = freq_S_indices[::subsampling]  # Subsample the frequency indices
    N = len(freq_S_indices)
    batch_size = batch_size

    for i in tqdm(range(0,N,batch_size),desc="Processing batches"):
        
        batch_end = min(i + batch_size, N) # Ensure we don't go past the end
        batch_slice = slice(i, batch_end)

        freq_indices_batch = freq_S_indices[batch_slice]
        array_man_batch = array_man[:,batch_slice]
        S_band_batch = S_band[:,:,freq_indices_batch]
        power_iq_batch = power_iq[batch_slice,0] if dataIq is not None else 1
        
        array_man_batch = array_man_batch * torch.sqrt(power_iq_batch)

        S_band_batch_array = S_band_batch * array_man_batch.unsqueeze(0)

        S_data_cube_batch = torch.einsum('bij,bkj->bik',S_band_batch_array.conj(), S_band_batch_array)
       
        sum_s = sum_s + S_data_cube_batch.cpu()

        if "cuda" in str(device):
            del S_data_cube_batch, S_band_batch_array #S_band_batch_array_conj
            gc.collect()
            torch.cuda.empty_cache()

    return sum_s


def calculate_reflected_power_directly(
    S_band,
    freq_S_indices,
    IQ_fft,
    array_man,
    broadsideWeights,
    Z_0,
    batch_size=2048, # Increased default batch size for better performance
    device="cpu"
):
    """
    Calculates the final reflected power directly in a memory-efficient way
    by fusing the logic and reformulating the math to avoid creating large
    intermediate tensors.

    Args:
        S_band (array-like): Shape (P, A, F), P=ports, A=antennas, F=freqs.
        freq_S_indices (array-like): Shape (N,). Indices into S_band's F-dim.
        IQ_fft (array-like): Shape (N,). FFT of the waveform.
        array_man (array-like): Shape (1, N, A). Array manifold drift.
        broadsideWeights (array-like): Shape (A,). Beamforming weights 'w'.
        Z_0 (float): Characteristic impedance.
        batch_size (int): Number of frequency steps to process at once.
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: A tensor of shape (P,) representing the reflected
                      power for each port.
    """
    # --- 1. Data Preparation and Sanity Checks ---
    print("Preparing tensors for memory-efficient calculation...")
    S_band = torch.asarray(S_band, dtype=torch.cfloat, device=device)
    IQ_fft = torch.asarray(IQ_fft, dtype=torch.cfloat, device=device)
    w = torch.asarray(broadsideWeights, dtype=torch.cfloat, device=device)
    
    # Squeeze and transpose array_man ONCE, outside the loop
    array_man_processed = torch.asarray(array_man, dtype=torch.cfloat, device=device).squeeze(-1).T

    if len(freq_S_indices) != len(IQ_fft):
        raise ValueError("freq_S_indices and IQ_fft must have the same length")
    
    N = len(freq_S_indices)
    n_ports = S_band.shape[0]

    # --- 2. Initialize a SMALL accumulator for the final result ---
    # The result is a vector of size P, which uses very little memory.
    total_reflected_power = torch.zeros(n_ports, device=device, dtype=torch.cfloat)

    print(f"Processing {N} signals in batches of {batch_size}...")

    for i in tqdm(range(0, N, batch_size),desc="Processing batches"):
        # --- 3. Get the current mini-batch ---
        batch_end = min(i + batch_size, N)
        batch_slice = slice(i, batch_end)
        
        freq_indices_batch = freq_S_indices[batch_slice]
        array_man_batch = array_man_processed[:, batch_slice]
        iq_fft_batch = IQ_fft[batch_slice,0]
        S_band_effective_batch = S_band[:, :, freq_indices_batch] * array_man_batch.unsqueeze(0)
        
        # This computes inner(S_band_effective[p,:,n], w) for all p and n
        # Shape: (P, N_batch)
        vH_w = torch.einsum('pan,a->pn', S_band_effective_batch, w)

        power_per_port_n = torch.real(vH_w.conj() * vH_w)
        
        # --- 5. Apply weights and accumulate the result for this batch ---
        weights_batch = torch.abs(iq_fft_batch)**2
        
        power_contribution = torch.einsum('pn,n->p', power_per_port_n, weights_batch)

        total_reflected_power += power_contribution

        del S_band_effective_batch, vH_w, power_per_port_n, weights_batch, power_contribution

    print("Calculation complete.")
    # Finally, divide by impedance. Use .real as power is a real quantity.
    return total_reflected_power.real / Z_0



def uv_coords_torch(theta_deg, phi_deg):
    theta = torch.deg2rad(theta_deg)
    phi = torch.deg2rad(phi_deg)
    u = torch.sin(theta) * torch.cos(phi)
    v = torch.sin(theta) * torch.sin(phi)
    return u, v

def check_safety_violation(S, theta1, phi1, theta2, phi2, device='cpu', batch_size=10):
    """
    S: (96, 96) complex coupling matrix
    theta*, phi*: (N,) float tensors in degrees
    Output: Boolean tensor of shape (N, N, N, N)
    """
    N1, N2, N3, N4 = len(theta1), len(phi1), len(theta2), len(phi2)
    total_elements = 96
    n_rows, n_cols = 4, 24
    split_col = 12

    # Move S to device
    S = torch.from_numpy(S).to(torch.cfloat).to(device)

    # Array indices
    y_idx, z_idx = torch.meshgrid(torch.arange(n_rows), torch.arange(n_cols), indexing='ij')
    y_flat = y_idx.T.reshape(-1).to(device)
    z_flat = z_idx.T.reshape(-1).to(device)
    col_mask = z_flat < split_col

    # Compute (u1, v1)
    theta1_rad = torch.deg2rad(theta1).to(device)
    phi1_rad = torch.deg2rad(phi1).to(device)
    u1 = torch.sin(theta1_rad).unsqueeze(1) * torch.cos(phi1_rad).unsqueeze(0)  # (N1, N2)
    v1 = torch.sin(theta1_rad).unsqueeze(1) * torch.sin(phi1_rad).unsqueeze(0)

    # Allocate final output
    violation = torch.zeros((N1, N2, N3, N4), dtype=torch.bool)
    array_man = torch.zeros((N1, N2, N3, N4,n_rows * n_cols), dtype=torch.cfloat)
    for i in range(0, N3, batch_size):
        for j in range(0, N4, batch_size):
            t2_batch = theta2[i:i+batch_size]
            p2_batch = phi2[j:j+batch_size]
            B3, B4 = len(t2_batch), len(p2_batch)

            theta2_rad = torch.deg2rad(t2_batch).to(device)
            phi2_rad = torch.deg2rad(p2_batch).to(device)
            u2 = torch.sin(theta2_rad).unsqueeze(1) * torch.cos(phi2_rad).unsqueeze(0)  # (B3, B4)
            v2 = torch.sin(theta2_rad).unsqueeze(1) * torch.sin(phi2_rad).unsqueeze(0)

            # Expand u1/v1 and u2/v2
            u1_exp = u1[:, :, None, None, None]
            v1_exp = v1[:, :, None, None, None]
            u2_exp = u2[None, None, :, :, None]
            v2_exp = v2[None, None, :, :, None]

                # Center y and z coordinates (static once)
            y_centered = torch.zeros(y_flat.shape, dtype=torch.float, device=device)
            z_centered = torch.zeros(z_flat.shape, dtype=torch.float, device=device)
            y_centered[...,col_mask] = y_flat[...,col_mask] - y_flat[...,col_mask].float().mean()
            z_centered[...,col_mask] = z_flat[...,col_mask] - z_flat[...,col_mask].float().mean()

            y_centered[...,~col_mask] = y_flat[...,~col_mask] - y_flat[...,~col_mask].float().mean()
            z_centered[...,~col_mask] = z_flat[...,~col_mask] - z_flat[...,~col_mask].float().mean()

            # Reshape for broadcasting
            y = y_centered.view(1, 1, 1, 1, -1)
            z = z_centered.view(1, 1, 1, 1, -1)
          

            # Construct manifold vector a (N1, N2, B3, B4, 96)
            a_all = torch.zeros((N1, N2, B3, B4, total_elements), dtype=torch.cfloat, device=device)
            a_all[..., col_mask] = torch.exp(1j * 2 * np.pi * ((y[..., col_mask]) * v1_exp + (z[..., col_mask]) * u1_exp))
            a_all[..., ~col_mask] = torch.exp(1j * 2 * np.pi * ((y[..., ~col_mask]) * v2_exp + (z[..., ~col_mask]) * u2_exp))

            # Reshape to flat: (P, 96)
            Sa = a_all @ S
            Sa_inf = Sa.abs().max(dim=-1)[0]  # (P,)

            # Check violation and reshape
            batch_violation = (Sa_inf > 1).cpu()
            violation[:, :, i:i+B3, j:j+B4] = batch_violation
            array_man[:, :, i:i+B3, j:j+B4] = a_all
    return violation, array_man


def get_cholesky(mat,device = "cpu"):
    mat = torch.asarray(mat,dtype=torch.cfloat,device=device)
    L,Q = torch.linalg.eigh(mat)
    L[L<=0] = 0
    chol = torch.einsum('bi,bij->bij', torch.sqrt(L), Q.mH)
    return chol.cpu().numpy()