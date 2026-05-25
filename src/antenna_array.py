import os
import numpy as np
import torch
import math
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Default data location. Override with ARRAY_SAFETY_DATA.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ARRAY_SAFETY_DATA",
    os.path.normpath(os.path.join(_HERE, "..", "data", "Data")),
)
_COUPLING_CACHE = {}


def _require_data_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "See release/data/README.md for instructions on fetching the "
            "coupling matrix and other external data."
        )
    return path


class Array:
    def __init__(self, arrayName="Vivaldi36", num_segments=1, segment_indices=None):
        """
        Initializes the Array object.

        Args:
            arrayName (str): The name of the array model ("Vivaldi36" or other).
            num_segments (int): The number of segments to divide the array into automatically.
                                This is ignored if segment_indices is provided. Defaults to 1.
            segment_indices (dict): A dictionary where keys are segment IDs (e.g., 0, 1, 2)
                                    and values are lists or arrays of element indices
                                    belonging to that segment.
        """
        self.arrayName = arrayName
        self.c = scipy.constants.c
        self.Z_0 = 50 # Ohms
        
        if self.arrayName =="Vivaldi36":
            self.nrows, self.ncols = 36, 36
        else:
            self.nrows, self.ncols = 4, 24
        
        self.nEl = self.nrows * self.ncols
        self.elLoc = self._generate_element_coords()
        self._get_coupling_matrix()
        
        # --- Segmentation Initialization ---
        self.num_segments = num_segments
        self.segment_indices = None  # Will be populated by the helper method
        self.segment_manifolds = {}  # Will store results after calculation
        self._initialize_segments(segment_indices)

        # Centering after creating the indices
        self.elLoc = self.elLoc - np.mean(self.elLoc)



    def _initialize_segments(self, segment_indices_manual):
        """
        Initializes the segment definitions based on manual or automatic input.
        """
        print(f"Initializing array with {self.num_segments} segment(s)...")
        if segment_indices_manual:
            print("Using manually provided segment indices.")
            self.segment_indices = segment_indices_manual
            self.num_segments = len(segment_indices_manual)
            return

        print("Automatically creating symmetric segments (column-wise split).")
        if self.num_segments <= 1:
            self.segment_indices = {0: np.arange(self.nEl)}
            return

        # Special case for Vivaldi36 with 4 segments (quadrant split)
        if self.arrayName == "Vivaldi36" and self.num_segments == 4:
            print("Automatically creating 4 symmetric quadrant segments for Vivaldi36.")
            self.segment_indices = {}
            index_grid = np.arange(self.nEl)

            index_bottom_left = index_grid[(self.elLoc.real < (5.7/39.3701)) & (self.elLoc.imag < (5.7/39.3701))]
            index_bottom_right = index_grid[(self.elLoc.real >= (5.7/39.3701)) & (self.elLoc.imag < (5.7/39.3701))]
            index_top_left = index_grid[(self.elLoc.real < (5.7/39.3701)) & (self.elLoc.imag >= (5.7/39.3701))]
            index_top_right = index_grid[(self.elLoc.real >= (5.7/39.3701)) & (self.elLoc.imag >= (5.7/39.3701))]

            self.segment_indices[0] = index_bottom_left
            self.segment_indices[2] = index_bottom_right    
            self.segment_indices[1] = index_top_left
            self.segment_indices[3] = index_top_right
            

            #self.plot_segment_layout(show_element_ids=True)
            return


        # Create segments by splitting the array column-wise
        if self.num_segments == 2:
            indices = np.arange(self.nEl)
            self.segment_indices  = {}
            
            indices_left = self.elLoc.real < (5.7/39.3701)
            
            self.segment_indices[0] = indices[indices_left]
            self.segment_indices[1] = indices[~indices_left]
        else:
            raise ValueError("Number of segments must be 1 or 2 or 4.")

        
        #self.plot_segment_layout(show_element_ids=True)
            
    def _generate_element_coords(self):
        if self.arrayName == "Vivaldi36":
            loc_path = _require_data_file('PC_xyz_m_36x36_Vivaldi.mat')
            loc_data = scipy.io.loadmat(loc_path)
            loc_data = loc_data['PC_xyz_m']
            elLoc = (loc_data[:,0] + 1j * loc_data[:,1])
        else:
            dx = 0.33 / 12 / 3.28
            cc, rr = np.meshgrid(np.arange(1, self.ncols + 1), np.arange(1, self.nrows + 1), indexing='ij')
            rr = rr.flatten()         
            cc = cc.flatten()
            elLoc = cc * dx - 1j * rr * dx

         # NOt centering it here since I want to break down the segments using absolute values
        return elLoc

    def calculate_array_manifold(self,freq_GHz,n_gridpoints=0,get_angle=True,theta_deg=0,phi_deg=0,device='cpu'):
        """
        Calculates array manifold(s) based on the operating mode.

        Args:
            freq_GHz (float or list/array): Center frequency or frequencies in GHz.
            n_gridpoints (int): Number of grid points for u/v scan (if get_angle=False).
            get_angle (bool): If True, calculates steering vector(s) for specific angle(s).
                              If False, calculates manifolds for a u-v grid scan.
            theta_deg (float or list): Angle(s) for steering vector calculation.
            phi_deg (float or list): Angle(s) for steering vector calculation.
            device (str): 'cpu' or 'cuda'.

        Returns:
            - dict: If get_angle is False, returns a dictionary of segment manifolds. The shape of each
                    manifold will be (num_freqs, num_segment_elements, num_grid_points).
            - np.ndarray: If get_angle is True, returns a composite steering vector. The shape will be
                          (num_freqs, num_total_elements).
        """
        freq = np.atleast_1d(freq_GHz) * 1e9
        n_freqs = len(freq)

        self.segment_manifolds = {}

        # --- Logic for get_angle=True: Composite Steering Vector ---
        if get_angle:
            theta_array = np.atleast_1d(theta_deg)
            phi_array = np.atleast_1d(phi_deg)
            if len(theta_array) != self.num_segments and len(theta_array) != 1:
                raise ValueError("Length of theta_deg must be 1 or equal to num_segments")
            if len(phi_array) != self.num_segments and len(phi_array) != 1:
                 raise ValueError("Length of phi_deg must be 1 or equal to num_segments")

            # Initialize for multiple frequencies
            composite_manifold = np.zeros((n_freqs, self.nEl), dtype=np.complex128)
            
            for seg_id, indices in self.segment_indices.items():
                theta = theta_array[0] if len(theta_array) == 1 else theta_array[seg_id]
                phi = phi_array[0] if len(phi_array) == 1 else phi_array[seg_id]
                
                u_seg = np.sin(theta * np.pi / 180) * np.cos(phi * np.pi / 180)
                v_seg = np.sin(theta * np.pi / 180) * np.sin(phi * np.pi / 180)

                segment_elLoc = self.elLoc[indices]
                segment_elLoc = segment_elLoc - np.mean(segment_elLoc)
                
                # Geometric phase term, shape: (n_segment_elements)
                geometric_phase = 2 * np.pi * (segment_elLoc.real * u_seg + segment_elLoc.imag * v_seg) / self.c
                
                # Use broadcasting to handle multiple frequencies
                # (n_freqs, 1) * (1, n_segment_elements) -> (n_freqs, n_segment_elements)
                exponent = 1j * freq[:, np.newaxis] * geometric_phase[np.newaxis, :]
                segment_manifold = np.exp(exponent)
                
                self.segment_manifolds[seg_id] = np.squeeze(segment_manifold)
                composite_manifold[:, indices] = segment_manifold
            
            return np.squeeze(composite_manifold)

        # --- Logic for get_angle=False: Dictionary of Grid Scans ---
        else:
            u_space = np.linspace(-1,1,n_gridpoints)
            v_space = np.linspace(-1,1,n_gridpoints)
            u,v = np.meshgrid(u_space,v_space,indexing='xy')
            u, v = u.flatten(), v.flatten()

            abs_u_v = np.sqrt(u**2 + v**2)
            u[abs_u_v > 1] = np.nan
            v[abs_u_v > 1] = np.nan

            
            for seg_id, indices in self.segment_indices.items():
                segment_elLoc = self.elLoc[indices]
                #segment_elLoc = segment_elLoc - np.mean(segment_elLoc)

                if n_freqs == 1:
                     segment_manifold = np.exp(1j * 2 * np.pi * (np.outer(segment_elLoc.real, u) 
                                                                 + np.outer(segment_elLoc.imag, v)) * (freq[0] / self.c))
                else:
                    n_grid_pts_flat = len(u)
                    segment_manifold = np.zeros((n_freqs, len(indices), n_grid_pts_flat), dtype=np.complex128)
                    for i in range(n_freqs):
                        segment_manifold[i,:,:] = np.exp(1j * 2 * np.pi * (np.outer(segment_elLoc.real, u)
                                                                            + np.outer(segment_elLoc.imag, v)) * (freq[i] / self.c))


                self.segment_manifolds[seg_id] = np.squeeze(segment_manifold)
            return self.segment_manifolds
    
    def _get_coupling_matrix(self):
        cache_key = self.arrayName
        cached = _COUPLING_CACHE.get(cache_key)
        if cached is not None:
            self.freq_S_Ghz = cached["freq_S_Ghz"].copy()
            self.Sf = cached["Sf"].copy()
            return

        if self.arrayName == "Vivaldi36":
            smat_path = _require_data_file('Smat_36x36_90MHz.mat')
            Smat_data = scipy.io.loadmat(smat_path)
            Smat = Smat_data['Smat']
            self.freq_S_Ghz= Smat['f_GHz'][0, 0].flatten()
            self.Sf = Smat['S'][0, 0]
        else:
            smat_path = _require_data_file('SimpleVivaldi_24x4_BruteForce18Pts.mat')
            Smat_data = scipy.io.loadmat(smat_path)
            Smat = Smat_data['Smat']
            self.freq_S_Ghz = Smat['f_GHz'][0, 0].flatten()
            self.freq_S_Ghz = np.asarray([3 + i/1e9 for i in range(len(self.freq_S_Ghz))])
            Sf = Smat['S'][0, 0]
            cc, rr = np.meshgrid(np.arange(1, self.ncols + 1), np.arange(1, self.nrows + 1), indexing='ij')
            rr, cc = rr.flatten(), cc.flatten()
            bandIdx = np.arange(0, len(self.freq_S_Ghz))
            idx = (rr - 1) * self.ncols + cc
            idx = idx.astype(int) - 1
            self.Sf = Sf[np.ix_(idx, idx, bandIdx)]

        self._enforce_passivity()
        _COUPLING_CACHE[cache_key] = {
            "freq_S_Ghz": self.freq_S_Ghz.copy(),
            "Sf": self.Sf.copy(),
        }

    def _enforce_passivity(self, tol=1e-3):
        Sf = np.asarray(self.Sf)
        max_violation = 0.0
        for k in range(Sf.shape[2]):
            s = np.linalg.svd(Sf[:, :, k], compute_uv=False)
            max_sigma = float(s[0])
            max_violation = max(max_violation, max_sigma - 1.0)
            if max_sigma > 1.0:
                U, s_full, Vh = np.linalg.svd(Sf[:, :, k], full_matrices=False)
                Sf[:, :, k] = (U * np.minimum(s_full, 1.0)) @ Vh
        self.Sf = Sf
        if max_violation > tol:
            print(f"[passivity] enforced; max sigma was {1+max_violation:.4f} (clamped to 1).")
    
    def get_center_freq_coupling_matrix(self,fc):
        # This method remains unchanged
        get_closest_freq_idx = np.argmin(np.abs(self.freq_S_Ghz - fc))
        return self.Sf[:,:,get_closest_freq_idx]
    
    def get_coupling_freq_idx(self,freq_of_data):
        # This method remains unchanged
        closest_S_freq_idx_to_data = np.array([np.argmin(np.abs(self.freq_S_Ghz  - f)) for f in freq_of_data])
        return closest_S_freq_idx_to_data

    def calculate_reflected_power_directly(self,
                            data,
                            array_man_freq,
                            broadsideWeights,
                            batch_size=2048, # Increased default batch size for better performance
                            device="cpu",
                            verbose = False,
                            no_S_flag = False,
                            return_cube = False):
        # --- 1. Data Preparation and Sanity Checks ---
        print("Preparing tensors for memory-efficient calculation...")
        IQ_fft = data.get_fft_signal()

        freq_signal = data.freq_signal
        
        S_band = torch.asarray(self.Sf, dtype=torch.cfloat, device=device)
        if return_cube:
            S_cube = torch.zeros(size=(self.nEl,self.nEl,self.nEl),dtype=torch.cfloat, device="cpu")
        else:
            S_cube = None
        if no_S_flag:
            S_band = torch.eye(self.nEl, dtype=torch.cfloat, device=device).unsqueeze(0).repeat(S_band.shape[0], 1, 1)
        IQ_fft = torch.asarray(IQ_fft, dtype=torch.cfloat, device=device)
        w = torch.asarray(broadsideWeights, dtype=torch.cfloat, device=device)
        # Normalize weights so ||w||^2 = 1 to conserve energy (||w||^2 = nEl for steering vectors)
        #w = w #/ torch.linalg.norm(w)

        freq_S_indices = self.get_coupling_freq_idx(freq_signal)

        if len(freq_S_indices) != len(IQ_fft):
            raise ValueError("freq_S_indices and IQ_fft must have the same length")

        N = len(freq_S_indices)

        # --- 2. Initialize accumulator for reflected power per port ---
        # Result: vector of size (P,) containing reflected power for each port
        total_reflected_power_per_port = torch.zeros(self.nEl, device=device, dtype=torch.float32)

        batch_size = min(N, batch_size)

        print(f"Processing {N} signals in batches of {batch_size}...")

        for i in tqdm(range(0, N, batch_size), desc="Processing batches"):
            # --- 3. Get the current mini-batch ---
            batch_end = min(i + batch_size, N)
            batch_slice = slice(i, batch_end)

            freq_indices_batch = freq_S_indices[batch_slice]
            iq_fft_batch = IQ_fft[batch_slice]

            # Get S-matrices for this batch: shape (P, P, N_batch)
            S_batch = S_band[:, :, freq_indices_batch].to(device)

            # Compute S_k × w for all frequencies in batch
            # Shape: (P, N_batch) where P is number of ports
            S_w = torch.einsum('pqn,q->pn', S_batch, w)

            # Compute |S_k × w|^2 for each port and frequency
            power_magnitude_sq = torch.abs(S_w) ** 2  # shape: (P, N_batch)

            # Get input signal power: |X(f_k)|^2
            # Shape: (N_batch,)
            input_signal_power = torch.abs(iq_fft_batch) ** 2

            # Compute power per port: Σ_n |S_k × w|_p^2 × |X(f_k)|^2
            # Sum over frequencies for each port: (P, N_batch) * (N_batch,) -> (P,)
            power_per_port_batch = torch.einsum('pn,n->p', power_magnitude_sq, input_signal_power.squeeze(-1))

            total_reflected_power_per_port += power_per_port_batch

            if return_cube:
                S_pw = S_batch * torch.sqrt(input_signal_power.view(1, 1, -1))
                S_cube += torch.einsum('prn,pqn->prq', S_pw, S_pw.conj()).cpu()

            del S_batch, S_w, power_magnitude_sq, input_signal_power, power_per_port_batch

        print("Calculation complete.")

        # Calculate total input power: Σ_k |X(f_k)|^2
        total_input_power = torch.sum(torch.abs(IQ_fft) ** 2).item()

        # Normalize: reflected coefficient squared per port = Σ_k |[S_k w]_p|^2 |X(f_k)|^2 / (|w_p|^2 * Σ_k |X(f_k)|^2)
        w_abs_sq = torch.abs(w) ** 2  # shape: (P,), real-valued
        reflected_coeff_per_port_square = total_reflected_power_per_port / (w_abs_sq * total_input_power)

        total_reflected_power_watts = torch.sum(total_reflected_power_per_port).item()
        loss_in_reflected_power = 10 * np.log10(total_reflected_power_watts / total_input_power / torch.sum(w_abs_sq).item())

        print(f"Loss from reflected power: {loss_in_reflected_power:.2f} dB")

        if verbose:
            max_reflection_coeff = torch.max(reflected_coeff_per_port_square).item()
            max_reflection_coeff_dB = 10 * np.log10(max_reflection_coeff)
            
            print(f"Max Reflection coefficient (per port): {max_reflection_coeff_dB:.4f} dB")
            

        return reflected_coeff_per_port_square, S_cube
    
    def plotArrayResponse(self,broadSide_weights,theta_deg,phi_deg,f_c,
                          n_gridpoints_per_u_v = 100,verbose = True,no_S_center=False,plot=False,vmin_db=None,vmax_db=None):
        
        '''
        Calculates and plots the array response, allowing for per-segment pointing directions.
        This method is adapted to the new outputs of calculate_array_manifold.
        '''
        # This method remains unchanged as it operates on a single frequency f_c
        theta_array = np.atleast_1d(theta_deg)
        phi_array = np.atleast_1d(phi_deg)

        if len(theta_array) == 1:
            print("⚠️ Warning: A single angle was provided. All segments will be pointed to this same direction.")
            theta_array = np.repeat(theta_array, self.num_segments)
            phi_array = np.repeat(phi_array, self.num_segments)
        elif len(theta_array) != self.num_segments:
            raise ValueError(f"The number of angles provided ({len(theta_array)}) does not match the number of segments ({self.num_segments}).")
        if len(phi_array) != self.num_segments:
             raise ValueError(f"The number of phi angles provided ({len(phi_array)}) does not match the number of theta angles.")

        segment_manifolds_scan = self.calculate_array_manifold(freq_GHz=f_c,get_angle=False,n_gridpoints=n_gridpoints_per_u_v)
        
        a_c_composite = self.calculate_array_manifold(freq_GHz=f_c, get_angle=True, theta_deg=theta_array, phi_deg=phi_array)
        
        S_center = self.get_center_freq_coupling_matrix(f_c)
        if no_S_center:
            S_center = np.eye(self.nEl, dtype=np.complex64)

        w_opt = a_c_composite * broadSide_weights
        w_effective = (np.eye(self.nEl, dtype=np.complex64)-S_center) @ w_opt
        
        segment_max_gains = {}

        global_max_gain = -np.inf
        temp_responses = {}
        for seg_id, indices in self.segment_indices.items():
            segment_A_scan = segment_manifolds_scan[seg_id]
            segment_w_effective = w_effective[indices]
            if segment_A_scan.size == 0 or len(indices) == 0: continue
            
            segment_response = (np.abs(np.conjugate(segment_A_scan).T @ segment_w_effective) / len(indices))
            segment_response_img = np.reshape(segment_response, (n_gridpoints_per_u_v, n_gridpoints_per_u_v)) # Reshape based on grid points
            segment_response_db = 10 * np.log10(np.abs(segment_response_img)**2)
            temp_responses[seg_id] = segment_response_db
            max_gain_segment = np.round(np.nanmax(segment_response_db), 2)
            if max_gain_segment > global_max_gain:
                global_max_gain = max_gain_segment
            segment_response_db = temp_responses[seg_id]
            max_gain_segment = np.round(np.nanmax(segment_response_db), 2)
            segment_max_gains[seg_id] = max_gain_segment

        if plot:
            nrows = int(np.ceil(np.sqrt(self.num_segments)))
            ncols = int(np.ceil(self.num_segments / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
            axes = axes.flatten()

            if vmax_db is not None and vmin_db is not None:
                # Use user-provided values if both are given
                vmax_plot = vmax_db
                vmin_plot = vmin_db
                print(f"Using user-provided color range: vmin={vmin_plot} dB, vmax={vmax_plot} dB")
            else:
                # Default behavior: calculate from data, ensuring vmax is not negative
                vmax_plot = max(0, global_max_gain)
                vmin_plot = vmax_plot - 60
            
            im = None
            for i, (seg_id, indices) in enumerate(self.segment_indices.items()):
                ax = axes[i]
                if seg_id not in temp_responses:
                    ax.set_title(f'Segment {seg_id} (No Data)')
                    continue
                
                #if i == 0:
                #    segment_response_db = np.flip(segment_response_db,axis=[0,1])
                #elif i==1:
                #    segment_response_db = np.flip(segment_response_db,axis=[1])
                #elif i == 2:
                #    segment_response_db = np.flip(segment_response_db,axis=[0,1])
                #elif i==3:
                #    segment_response_db = np.flip(segment_response_db,axis=[1])
                
                im = ax.imshow(segment_response_db, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='viridis', vmin=vmin_plot, vmax=vmax_plot)
                
                ax.set_xlabel('u-axis')
                ax.set_ylabel('v-axis')
                ax.set_title(f'Segment {seg_id} @ ({theta_array[i]}°, {phi_array[i]}°) | Max Gain: {max_gain_segment} dB')

                u = np.sin(theta_array[i] * np.pi / 180) * np.cos(phi_array[i] * np.pi / 180)
                v = np.sin(theta_array[i] * np.pi / 180) * np.sin(phi_array[i] * np.pi / 180)
                ax.scatter(u, v, 120, color='red', marker='x')
                
                theta_circle = np.linspace(0, 2 * np.pi, 361)
                ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k', lw=0.5, alpha=0.7)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            if im: # Only draw colorbar if at least one plot was made
                fig.subplots_adjust(right=0.85) # Adjust subplot to make room for colorbar
                cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7]) # Create a new axis for the colorbar
                fig.colorbar(im, cax=cbar_ax, label='Response (dB)')
            
            plt.suptitle('Array Response by Segment', fontsize=16)
            
            plt.show()

        if verbose:
            print(f"Segment Max Gains (dB): {segment_max_gains}")
        
        return segment_max_gains
    
    def plot_segment_layout(self, figsize=(8, 8), marker_size=50, show_element_ids=False):
        """
        Plots the physical layout of the array elements, coloring each segment differently.

        Args:
            figsize (tuple): The size of the matplotlib figure.
            marker_size (int): The size of the markers for each element.
            show_element_ids (bool): If True, annotates each element with its index number.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use a warm colormap like 'autumn' to get distinct colors for each segment
        colors = cm.get_cmap('autumn', self.num_segments)

        for seg_id, indices in self.segment_indices.items():
            # Get the physical locations for the elements in the current segment
            segment_locs = self.elLoc[indices]
            
            # Plot the elements with a unique color and label
            ax.scatter(
                segment_locs.real, 
                segment_locs.imag, 
                s=marker_size, 
                color=colors(seg_id), 
                label=f'Segment {seg_id}'
            )
            
            # If requested, add text labels for each element index
            if show_element_ids:
                for i, element_index in enumerate(indices):
                    loc = segment_locs[i]
                    ax.text(loc.real, loc.imag, str(element_index), fontsize=6, ha='center', va='center')

        ax.set_title('Array Segment Layout')
        ax.set_xlabel('X-coordinate (meters)')
        ax.set_ylabel('Y-coordinate (meters)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)
        if self.num_segments > 1:
            ax.legend()
        
        plt.show()
