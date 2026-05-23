import numpy as np
import scipy
import torch
import cvxpy as cp
import autograd.numpy as anp
import utils
import pymanopt
import h5py
import time
import mosek.fusion as mf
from tqdm import tqdm

EPS = 1e-8

class SolveArraySafety():
    def __init__(self,S_c,a_c,reg=1.0,device = "cpu",verbose=False, gmin = None):

        self.device = device
        self.reg = reg
        self.nEl = S_c.shape[0]
        self.S_c = S_c
        self.a_c = a_c
        self.verbose = verbose
        self._I_minus_S = np.eye(self.nEl, dtype=np.complex128) - self.S_c
        self.a_tilda = self.a_c.conj().T @ self._I_minus_S
        if gmin is not None:
            self.gmin = gmin
        else:
            self.gmin = self.nEl // 2
        # total signal energy Sum_k |X(f_k)|^2 -- needed by the iteration
        # recorder to convert max_val into true dB-Gamma^2 = max_val / total_input_power.
        # Default 1.0 reproduces the legacy behavior; callers should set this
        # via `solver_obj.total_input_power = float(np.sum(np.abs(X)**2))`
        # immediately after construction when running iteration sweeps.
        self.total_input_power = 1.0
        self._M = None  # MOSEK model, created lazily

    def dispose(self):
        """Dispose of MOSEK model to free native (non-Python) memory."""
        if self._M is not None:
            try:
                self._M.dispose()
            except Exception:
                pass
            self._M = None

    def update_steering(self, a_c_new, gmin=None):
        """Update steering vector without rebuilding MOSEK model.

        The MOSEK model structure depends only on nEl, so it can be
        reused across different steering directions.
        """
        self.a_c = a_c_new
        self.a_tilda = self.a_c.conj().T @ self._I_minus_S
        if gmin is not None:
            self.gmin = gmin
        # Update torch/numpy versions used in update_weights_dual
        self.a_tilda_torch_cpu = torch.from_numpy(np.conj(self.a_tilda).T).to(torch.complex64)
        self._coeffs_np_cvx = self.a_tilda_torch_cpu.numpy().reshape(-1, 1)

    def solveArraySafety_inf(self, S_f, freq_S_in_signal_idx, sub_sample=1, data_iq=None):
        """
        Minimizes max per-port reflected power (brute force, all ports).
        Computes per-port Gram matrices via batched BLAS GEMM, then solves full SOCP.
        """
        K = len(freq_S_in_signal_idx)
        if data_iq is None:
            data_power = np.ones(K)
        else:
            data_power = np.abs(data_iq.get_fft_signal()) ** 2

        s_idx_raw = np.asarray(freq_S_in_signal_idx[::sub_sample]).ravel()
        pwr_raw = np.asarray(data_power[::sub_sample][:len(s_idx_raw)]).ravel()

        # Deduplicate: many signal freqs map to the same S-param freq.
        # Sum their power weights so the GEMM inner dimension shrinks from K to n_unique.
        unique_s_idx, inverse = np.unique(s_idx_raw, return_inverse=True)
        pwr_unique = np.bincount(inverse.ravel(), weights=pwr_raw, minlength=len(unique_s_idx))
        pwr_sqrt = np.sqrt(pwr_unique).astype(np.float64)
        n_unique = len(unique_s_idx)

        print(f"Building {self.nEl} factor matrices D_m "
              f"(nEl={self.nEl}, K_raw={len(s_idx_raw)}, K_unique={n_unique})...")
        t0 = time.time()

        # Instead of forming Q_m = D_m^H @ D_m (nEl x nEl) and then having CVXPY
        # re-decompose it, keep the thin factor D_m (n_unique x nEl) directly.
        # Constraint: ||D_m @ w||^2 <= t  →  SOC of dimension n_unique, NOT nEl.
        # This makes the MOSEK problem ~10-25x smaller.
        D_list = [None] * self.nEl

        for m in range(self.nEl):
            # D_m: (n_unique, nEl) — row k = S_f[m, :, unique_s_idx[k]] * sqrt(pwr[k])
            D_list[m] = S_f[m, :, unique_s_idx] * pwr_sqrt[:, None]

        print(f"Factor matrices built in {time.time() - t0:.1f}s. Building CVXPY problem...")
        t0 = time.time()

        w = cp.Variable((self.nEl, 1), complex=True)
        t_var = cp.Variable(1, nonneg=True)

        # ||D_m @ w||^2 = w^H (D_m^H D_m) w = w^H Q_m w  — same math, smaller SOC
        constraints = [cp.sum_squares(D_list[m] @ w) <= t_var
                       for m in range(self.nEl)]

        gain_coeffs = self.a_c.conj().T @ self._I_minus_S
        constraints.append(cp.real(gain_coeffs @ w) >= self.gmin)
        constraints.append(cp.norm(w, 'inf') <= 1)

        prob = cp.Problem(cp.Minimize(t_var), constraints)
        print(f"CVXPY problem built in {time.time() - t0:.1f}s. Solving with MOSEK...")

        prob.solve(solver=cp.MOSEK, verbose=self.verbose)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Solver status: {prob.status}")

        return w.value, None

    
    def solveArraySafety(self,S_data_cube):
        
        S_data_cube = utils.get_nearest_psd_matrix(S_data_cube)
        S_data_cube = np.asarray(S_data_cube, dtype=np.complex64)
        
        # Define the CVXPY variables
        w = cp.Variable((self.nEl, 1), complex=True, name='w')
        epsilon = cp.Variable(1, nonneg=True, name='epsilon')

        # 2. FIX: Introduce a slack variable 't' for the epigraph formulation
        t = cp.Variable(1, name='t')

        # The objective is now linear and DCP-compliant
        objective = cp.Minimize(t + self.reg * epsilon)

        # Build the list of power expressions
        power_expressions = []
        for i in range(S_data_cube.shape[0]):
            power_expressions.append(cp.quad_form(w, S_data_cube[i, :, :] + 1e-6))

        constraints = [cp.vstack(power_expressions) <= t]

        # Add your original constraint
        coeffs = np.conj(self.a_tilda) 
        term_scalar_product = coeffs.T @ w
        constraints.append(cp.real(term_scalar_product) >= 1 - epsilon)


        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=self.verbose)

        # Optimal weights
        w_opt = w.value
   
        return w_opt, epsilon.value
    
    def solverArraySafety_phaseOnly(self,S_data_cube,thresh = 1e-9, max_iter = 1e4, debug = True, device = "cuda",
                                     mosek_tol_start=1e-3, mosek_tol_end=1e-6, mosek_max_iter=100,
                                     w_init=None, precomputed_alpha=None, precomputed_conj=None,mosek_direct=True,
                                     progress_every=None, progress_label="iMM"):

        # --- 1. Setup: determine optimal data placement and pre-compute ---
        device = torch.device(device)
        self.device = device
        self.thresh = thresh
        self.debug = debug
        self.mosek_tol_start = mosek_tol_start
        self.mosek_tol_end = mosek_tol_end
        self.mosek_max_iter = mosek_max_iter
        num_slices = S_data_cube.shape[0]

        # Reuse precomputed conj tensor if provided (same data cube across runs)
        if precomputed_conj is not None:
            self.s_data_cube_tensor = precomputed_conj.conj()  # store original
            self.s_data_cube_conj = precomputed_conj
            self._data_on_gpu = precomputed_conj.device.type == 'cuda'
        else:
            self.s_data_cube_tensor = S_data_cube.to(torch.complex64)
            self._data_on_gpu = False

            if device.type == 'cuda':
                data_bytes = self.s_data_cube_tensor.nelement() * self.s_data_cube_tensor.element_size()
                gpu_total = torch.cuda.get_device_properties(device).total_memory
                if data_bytes < gpu_total * 0.6:
                    self.s_data_cube_tensor = self.s_data_cube_tensor.to(device)
                    self._data_on_gpu = True
                    print(f"Data cube ({data_bytes / 1e9:.1f} GB) fits on GPU — GPU-resident mode.")
                else:
                    self.s_data_cube_tensor = self.s_data_cube_tensor.pin_memory()
                    self._gpu_batch_size = max(1, int(gpu_total * 0.25 / (self.nEl * self.nEl * 8)))
                    print(f"Data cube ({data_bytes / 1e9:.1f} GB) pinned on CPU — GPU batch size: {self._gpu_batch_size}.")

            self.s_data_cube_conj = self.s_data_cube_tensor.conj()

        # Reuse precomputed eigenvalues if provided, otherwise compute them.
        # alpha must live on the same device as the cube, otherwise the pure-CPU
        # path in _compute_C_j_B_j hits a device mismatch when the cube is too
        # large to fit on the GPU.
        alpha_device = self.s_data_cube_conj.device
        if precomputed_alpha is not None:
            self.alpha = precomputed_alpha.to(alpha_device)
        else:
            print(f"Calculating {num_slices} eigenvalues...")
            self.alpha = torch.max(torch.linalg.eigvalsh(self.s_data_cube_conj), dim=1).values
            print("Eigenvalue calculation complete.")

        # --- 2. Pre-compute constants ---
        self.a_tilda_torch_cpu = torch.from_numpy(np.conj(self.a_tilda).T).to(torch.complex64)
        self._coeffs_np_cvx = self.a_tilda_torch_cpu.numpy().reshape(-1, 1)

        # Pre-allocate buffers for MOSEK parameter updates (avoids allocation per iteration)
        n = self.nEl + 1
        self._B_buf = np.empty((self.nEl, n), dtype=np.complex128)
        self._B_buf[:, -1] = (-self._coeffs_np_cvx).ravel()
        self._c_buf = np.empty(n, dtype=np.float64)
        self._c_buf[-1] = self.gmin
        self._B_re_buf = np.empty((self.nEl, n), dtype=np.float64)
        self._B_im_buf = np.empty((self.nEl, n), dtype=np.float64)

        if self._M is None:
            self._build_mosek_problem()

        # Use warm-start initial weights if provided, otherwise default to a_c
        if w_init is not None:
            w_gpu = w_init.to(torch.complex64)
        else:
            w_gpu = torch.asarray(self.a_c).to(torch.complex64)

        if self.debug:
            self.record_obj_value = []
            self.dual_value = []
            self.primal_value = []
            self.mu = []

        _t0 = time.time()
        for iter in range(int(max_iter)):
            # update_weights_dual records Gamma^2 of the CURRENT (pre-update) w
            # via _compute_C_j_B_j -> C_j -> wHSw. So record_obj_value[k] is
            # the value at iterate k (k=0 is the initial conjugate weights).
            w_gpu = self.update_weights_dual(w_gpu, num_slices, iter)
            if progress_every and ((iter + 1) % progress_every == 0 or iter == 0):
                _elapsed = time.time() - _t0
                _rate = (iter + 1) / _elapsed if _elapsed > 0 else 0.0
                _remain = (int(max_iter) - (iter + 1)) / _rate if _rate > 0 else float('inf')
                _last = self.record_obj_value[-1] if getattr(self, 'record_obj_value', None) else float('nan')
                print(f"[{progress_label}] iter {iter+1}/{int(max_iter)}  "
                      f"obj={_last:+.4f} dB  rate={_rate:.2f} it/s  "
                      f"elapsed={_elapsed:.0f}s  eta={_remain:.0f}s",
                      flush=True)

        # Append one final value for the post-last-update weights so length
        # matches solver_FASTA's recorded series (max_iter + 1 entries).
        if self.debug:
            _final_max, _ = self._get_inf_norm_obj_value_ram_gpu(w_gpu)
            self.record_obj_value.append(
                10.0 * np.log10(_final_max * 50.0 * self.nEl / self.total_input_power)
            )

        return w_gpu, getattr(self, 'record_obj_value', None), getattr(self, 'dual_value', None), getattr(self, 'primal_value', None), getattr(self, 'mu', None)

    def solve_iMM_sweep(self, S_data_cube, s_data_cube_list=None, thresh=1e-9, max_iter=1e4, debug=False,
                        device="cuda", mosek_tol_start=1e-3, mosek_tol_end=1e-6,
                        mosek_max_iter=100):
        """
        Runs iMM for 500 UV sweep points with warm-starting and shared precomputation.

        Eigenvalues, conjugate tensor, and device placement are computed ONCE from
        S_data_cube (which is the same for all points). The DPP problem is also
        built once and reused.

        Args:
            S_data_cube: torch.Tensor of shape (num_slices, nEl, nEl). The shared
                         coupling data cube (same for all sweep points).
            s_data_cube_list: ignored (kept for backward compat). S_data_cube is
                              reused for every point since it doesn't change.
            thresh, max_iter, debug, device: same as solverArraySafety_phaseOnly.
            mosek_tol_start, mosek_tol_end, mosek_max_iter: MOSEK adaptive tolerance params.

        Returns:
            results: list of dicts with keys 'w_opt' (and 'record_obj_value' etc. if debug=True).
            total_time: total wall-clock time in seconds.
        """
        device = torch.device(device)
        num_points = len(s_data_cube_list) if s_data_cube_list is not None else 1

        # --- Precompute ONCE: device placement, conj, eigenvalues ---
        print("Precomputing shared data (device placement, conjugate, eigenvalues)...")
        s_tensor = S_data_cube.to(torch.complex64)
        
        if device.type == 'cuda':
            data_bytes = s_tensor.nelement() * s_tensor.element_size()
            gpu_total = torch.cuda.get_device_properties(device).total_memory
            if data_bytes < gpu_total * 0.6:
                s_tensor = s_tensor.to(device)
                _data_on_gpu = True
            else:
                s_tensor = s_tensor.pin_memory()
                self._gpu_batch_size = max(1, int(gpu_total * 0.25 / (self.nEl * self.nEl * 8)))

        s_conj = s_tensor.conj()
        num_slices = s_tensor.shape[0]

        print(f"Computing {num_slices} eigenvalues (once)...")
        alpha = torch.max(torch.linalg.eigvalsh(s_conj), dim=1).values
        print("Eigenvalue precomputation complete.")

        # --- Run the sweep ---
        results = []
        w_prev = None

        t_start = time.time()
        for idx in tqdm(range(num_points), desc="iMM sweep"):
            w_opt, rec_obj, dual_val, primal_val, mu_val = self.solverArraySafety_phaseOnly(
                S_data_cube,
                thresh=thresh,
                max_iter=max_iter,
                debug=debug,
                device=device,
                mosek_tol_start=mosek_tol_start,
                mosek_tol_end=mosek_tol_end,
                mosek_max_iter=mosek_max_iter,
                w_init=w_prev,
                precomputed_alpha=alpha,
                precomputed_conj=s_conj,
            )

            result = {'w_opt': w_opt}
            if debug:
                result['record_obj_value'] = rec_obj
                result['dual_value'] = dual_val
                result['primal_value'] = primal_val
                result['mu'] = mu_val

            results.append(result)
            w_prev = w_opt.clone()

        total_time = time.time() - t_start
        print(f"iMM sweep complete: {num_points} points in {total_time:.1f}s "
              f"({total_time/num_points:.2f}s/point)")

        return results, total_time

    def _build_dual_problem(self):
        """Build the CVXPY dual problem once using Parameters (DPP).

        This avoids re-parsing and re-compiling the problem structure every
        iteration. Only the parameter values are updated before each solve.
        """
        self._C_j_param = cp.Parameter(shape=(self.nEl, 1), name='C_j')
        self._B_j_T_param = cp.Parameter(shape=(self.nEl, self.nEl), complex=True, name='B_j_T')

        self._lambda1_var = cp.Variable(shape=(self.nEl, 1), nonneg=True, name='lambda1')
        self._mu_var = cp.Variable(1, nonneg=True, name='mu')

        objective = cp.Maximize(
            cp.real(self._lambda1_var.T @ self._C_j_param)
            - cp.norm1(self._B_j_T_param @ self._lambda1_var - self._mu_var * self._coeffs_np_cvx)
            + self._mu_var * self.gmin
        )
        constraints = [cp.sum(self._lambda1_var) == 1]
        self._dual_prob = cp.Problem(objective, constraints)

    def _build_mosek_problem(self):
        """
        Builds the MOSEK Fusion model once using Parameters.
        Maintains exact structure of solve_mosek_direct.
        """
        # Dispose old model first to avoid native memory leak
        self.dispose()

        m = self.nEl
        n = self.nEl + 1

        self._M = mf.Model("l1_simplex_compiled")
        self._M.acceptedSolutionStatus(mf.AccSolutionStatus.Anything)

        # --- Parameters (Replacements for static data) ---
        self._B_re_param = self._M.parameter("B_re", [m, n])
        self._B_im_param = self._M.parameter("B_im", [m, n])
        self._c_param = self._M.parameter("c", n)

        # --- Variables (Exact match) ---
        x = self._M.variable("x", n-1, mf.Domain.greaterThan(0.0))
        mu = self._M.variable("mu", 1, mf.Domain.greaterThan(0.0))
        x_stack = mf.Expr.vstack(x, mu)
        t = self._M.variable("t", m, mf.Domain.unbounded())

        # --- Logic (Exact match) ---
        # Use Parameters instead of mf.Matrix.dense
        z_re = mf.Expr.mul(self._B_re_param, x_stack)
        z_im = mf.Expr.mul(self._B_im_param, x_stack)

        t_col = mf.Expr.reshape(t, m, 1)
        z_re_col = mf.Expr.reshape(z_re, m, 1)
        z_im_col = mf.Expr.reshape(z_im, m, 1)

        # Stack Horizontally -> Shape (m, 3)
        cone_input = mf.Expr.hstack(t_col, z_re_col, z_im_col)

        self._M.constraint("cones", cone_input, mf.Domain.inQCone().axis(1))
        
        # sum(x) == 1
        self._M.constraint("simplex", mf.Expr.sum(x), mf.Domain.equalsTo(1.0))

        # Objective: min 1^T t - c^T x
        ones_m = [1.0] * m
        obj = mf.Expr.sub(mf.Expr.dot(ones_m, t), mf.Expr.dot(self._c_param, x_stack))
        self._M.objective("obj", mf.ObjectiveSense.Minimize, obj)

        # --- Performance tuning ---
        # Default MOSEK tolerance is 1e-8 which is overkill for an iMM inner
        # subproblem. Loosening to 1e-6 and disabling presolve cut per-iter
        # solve time ~20% on the Vivaldi-36 case (profile_mosek_speedups.py).
        try:
            self._M.setSolverParam("intpntCoTolPfeas", 1e-6)
            self._M.setSolverParam("intpntCoTolDfeas", 1e-6)
            self._M.setSolverParam("intpntCoTolRelGap", 1e-6)
            self._M.setSolverParam("numThreads", 8)
            self._M.setSolverParam("presolveUse", "off")
        except Exception:
            pass



    def _compute_C_j_B_j(self, w):
        """Compute C_j and B_j, leveraging GPU when available.

        Math (Q_j = S_j^* - alpha_j I):
            C_j = Re(-w^H S_j^* w) + 2 * alpha_j * nEl
            B_j = 2 * S_j^* @ w - 2 * alpha_j * w
        """
        data_device = self.s_data_cube_conj.device
        num_slices = self.s_data_cube_conj.shape[0]

        with torch.no_grad():
            if data_device.type == 'cuda':
                # Data on GPU — single vectorized pass
                w_dev = w.to(data_device)
                w_conj = w_dev.conj()
                Sw = torch.einsum('bjk,k->bj', self.s_data_cube_conj, w_dev)
                wHSw = torch.einsum('j,bj->b', w_conj, Sw)
                C_j = (torch.real(-wHSw) + 2 * self.alpha * self.nEl).cpu()
                B_j = (2 * Sw - 2 * self.alpha.unsqueeze(1) * w_dev.unsqueeze(0)).cpu()

            elif hasattr(self, '_gpu_batch_size') and self.device.type == 'cuda':
                # Data pinned on CPU — compute in GPU batches
                w_gpu = w.to(self.device)
                w_conj = w_gpu.conj()
                batch_size = self._gpu_batch_size
                C_j = torch.empty(num_slices, dtype=torch.float32)
                B_j = torch.empty(num_slices, self.nEl, dtype=torch.complex64)

                for i in range(0, num_slices, batch_size):
                    end = min(i + batch_size, num_slices)
                    S_batch = self.s_data_cube_conj[i:end].to(self.device, non_blocking=True)
                    alpha_batch = self.alpha[i:end].to(self.device, non_blocking=True)
                    wHSw = torch.einsum('j,bjk,k->b', w_conj, S_batch, w_gpu)
                    Sw = torch.einsum('bjk,k->bj', S_batch, w_gpu)
                    C_j[i:end] = (torch.real(-wHSw) + 2 * alpha_batch * self.nEl).cpu()
                    B_j[i:end] = (2 * Sw - 2 * alpha_batch.unsqueeze(1) * w_gpu.unsqueeze(0)).cpu()

            else:
                # Pure CPU path
                w_cpu = w.cpu() if w.device.type != 'cpu' else w
                w_conj = w_cpu.conj()
                Sw = torch.einsum('bjk,k->bj', self.s_data_cube_conj, w_cpu)
                wHSw = torch.einsum('j,bj->b', w_conj, Sw)
                C_j = torch.real(-wHSw) + 2 * self.alpha * self.nEl
                B_j = 2 * Sw - 2 * self.alpha.unsqueeze(1) * w_cpu.unsqueeze(0)

        return C_j, B_j
    
    

    def update_weights_dual(self, w, num_slices, outer_iter=0, direct_mosek=True):
        """iMM dual update step using vectorized C_j/B_j (CPU) and parameterized MOSEK/CVXPY."""

        C_j, B_j = self._compute_C_j_B_j(w)
        B_j_T = B_j.T

        # Record the ACTUAL per-port max Gamma^2 at the CURRENT iterate w (the
        # input w, before this MOSEK update). This matches the metric used by
        # solver_FASTA's recorder and by calculate_max_reflected_power, so the
        # iMM/PsGM/RsGM curves all share the same y-axis. wHSw[p] is recovered
        # from C_j and alpha at zero extra cost (the einsum is already done in
        # _compute_C_j_B_j); the previous recorder used the MOSEK surrogate
        # value, which sits ABOVE the true f and starts at the wrong point.
        if self.debug:
            alpha_cpu = self.alpha.detach().cpu().numpy() if hasattr(self.alpha, 'is_cuda') else np.asarray(self.alpha)
            wHSw_max_curr = float((-C_j.numpy() + 2.0 * alpha_cpu * self.nEl).max())
            if wHSw_max_curr > 0:
                self.record_obj_value.append(10.0 * np.log10(wHSw_max_curr / self.total_input_power))
            else:
                self.record_obj_value.append(float('nan'))
        C_j_np = C_j.numpy().astype(np.float64)
        B_j_T_np = B_j_T.numpy().astype(np.complex128)

        if direct_mosek:
            # Fill pre-allocated buffers (last column/element already set)
            self._B_buf[:, :-1] = B_j_T_np
            self._c_buf[:-1] = C_j_np

            # Guard against NaN/Inf propagating into MOSEK
            if not (np.all(np.isfinite(self._B_buf)) and np.all(np.isfinite(self._c_buf))):
                return w

            # Decompose into real/imag using pre-allocated buffers
            np.copyto(self._B_re_buf, self._B_buf.real)
            np.copyto(self._B_im_buf, self._B_buf.imag)

            self._B_re_param.setValue(self._B_re_buf)
            self._B_im_param.setValue(self._B_im_buf)
            self._c_param.setValue(self._c_buf)

            try:
                self._M.solve()
                lambda_opt = np.array(self._M.getVariable("x").level())
                mu_opt_val = np.array(self._M.getVariable("mu").level())[0]
                obj_value = -1 * self._M.primalObjValue()
            except Exception:
                # Model may be in a bad state — rebuild and retry once
                self._build_mosek_problem()
                self._B_re_param.setValue(self._B_re_buf)
                self._B_im_param.setValue(self._B_im_buf)
                self._c_param.setValue(self._c_buf)
                try:
                    self._M.solve()
                    lambda_opt = np.array(self._M.getVariable("x").level())
                    mu_opt_val = np.array(self._M.getVariable("mu").level())[0]
                    obj_value = -1 * self._M.primalObjValue()
                except Exception:
                    return w  # give up this iteration, keep current weights

            # Validate solution before using it
            if not (np.all(np.isfinite(lambda_opt)) and np.isfinite(mu_opt_val)):
                return w

            lambda1_opt = torch.asarray(lambda_opt, dtype=torch.cfloat)

        else:
            # Adaptive tolerance: start loose, tighten as outer loop converges
            # Log-space interpolation from mosek_tol_start -> mosek_tol_end over ~20 iters
            self._C_j_param.value = C_j_np.reshape(-1, 1)
            self._B_j_T_param.value = B_j_T_np
            progress = min(outer_iter / 20.0, 1.0)
            current_tol = self.mosek_tol_start * (self.mosek_tol_end / self.mosek_tol_start) ** progress

            
            self._dual_prob.solve(
                solver=cp.MOSEK, verbose=self.verbose, warm_start=True,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': current_tol,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': current_tol,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': current_tol,
                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': self.mosek_max_iter,
                }
            )
        
            # --- 3. Extract optimal values ---
            mu_opt_val = self._mu_var.value[0] if self._mu_var.value is not None else 0.0
            obj_value = self._dual_prob.value
            lambda1_opt = torch.asarray(self._lambda1_var.value, dtype=torch.float32).squeeze()

        if self.debug:
            self.mu.append(mu_opt_val)


        angle_arg = -B_j_T @ lambda1_opt + mu_opt_val * self.a_tilda_torch_cpu
        w_opt = torch.exp(1j * torch.angle(angle_arg))

        return w_opt
        
    
    
    def solverArraySafety_phaseOnly_ADMM(self,S_data_cube,thresh = 1e-6, debug = True):

        # Inititalization for this method
        self.thresh = thresh
        self.S_data_cube = utils.get_nearest_psd_matrix(S_data_cube)
        self.S_data_cube = np.asarray(self.S_data_cube, dtype=np.complex128)

        self.a_tilda_torch = np.asarray(self.a_tilda, dtype=np.complex128)
        self.a_c_torch  = np.asarray(self.a_c, dtype=np.complex128)

        # Begin ADMM iterations
        w = np.ones(shape=(self.nEl,1), dtype=np.complex128)
        y = np.zeros(shape=(self.nEl,1), dtype=np.complex128)
        self.rho=1
        self.mu = 5
        max_iter = 20
        iter = 0
        cond = True
        obj_value = torch.inf

        if debug:
            record_obj_value = []
        while((cond or iter < 5) and iter < max_iter):
            prev_obj_value = obj_value
            obj_value = self.get_inf_norm_obj_value(w)
            record_obj_value.append(obj_value) if debug else None
            # Update w
            w,y,z, primal_residual, dual_residual = self.update_weights_ADMM(w,y)

            # Check convergence
            primal_tol = np.sqrt(self.nEl) * 1e-1 + 1e-4 * np.max([np.linalg.norm(z),np.sqrt(self.nEl)])
            dual_tol = np.sqrt(self.nEl) * 1e-1 + 1e-4 * np.linalg.norm(y)

            cond = primal_residual > primal_tol or dual_residual > dual_tol
            
            if iter > 1:
                if obj_value > prev_obj_value :
                    assert "Objective value increased, something is wrong!"
                    
            iter += 1
        
        return w, record_obj_value if debug else None
    

    def update_weights_ADMM(self,w,y):

        # Solving cvxpy problem for z 
        z = cp.Variable(shape=(self.nEl,1), complex=True)
        epsilon = cp.Variable(1, nonneg=True, name='epsilon')

        # 2. FIX: Introduce a slack variable 't' for the epigraph formulation
        t = cp.Variable(1, name='t')

        # The objective is now linear and DCP-compliant
        objective = cp.Minimize(t + self.reg * epsilon + cp.real(y.conj().T @ z) + (self.rho/2) * cp.norm2(z - w)**2)

        # Build the list of power expressions
        power_expressions = []
        for i in range(self.S_data_cube.shape[0]):
            power_expressions.append(cp.quad_form(z, self.S_data_cube[i, :, :]))

        constraints = [cp.vstack(power_expressions) <= t]

        # Add your original constraint
        coeffs = np.conj(self.a_tilda) * self.a_c
        term_scalar_product = coeffs.T @ z
        constraints.append(cp.real(term_scalar_product) >= 1 - epsilon)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=self.verbose)

        z_opt = np.asarray(z.value, dtype=np.complex128)

        # Update w
        w_new = np.exp(1j * np.angle(self.rho * z_opt + y))

        # Update y
        y_new = y + self.rho * (w_new - z_opt)

        # Updating the step size 
        primal_residual = np.linalg.norm(w_new - z_opt)
        dual_residual = np.linalg.norm(self.rho * (w - w_new))

        if primal_residual > self.mu * dual_residual:
            self.rho *= 2
        elif dual_residual > self.mu * primal_residual:
            self.rho /= 2
        
        return w_new, y_new, z_opt, primal_residual, dual_residual
    


    def solver_manopt(self,S_f,a_look_f,freq_S_in_signal_idx,sub_sample = 250, data_iq = None):
        K = len(freq_S_in_signal_idx)
        C_list = []
        if data_iq is None:
            data_power = np.ones(K)
        else:
            data_power = np.abs(data_iq.get_fft_signal()) ** 2

        for i in range(0,K,sub_sample):
            C_i = S_f[:, :, freq_S_in_signal_idx[i]] @ np.diag(a_look_f[i, :]) * data_power[i]
            C_list.append(C_i)
        C_stack = anp.vstack(C_list)
        dim = int(np.ceil(K / sub_sample))
        coeffs = self.a_c.conj().T @ ((np.eye(self.nEl,dtype = np.complex128) - self.S_c) @ np.diag(self.a_c))

        manifold = pymanopt.manifolds.complex_circle.ComplexCircle(self.nEl)

        # Constraint violation penalty
        mu = 0
        @pymanopt.function.autograd(manifold)
        def cost(w):
            power_terms = anp.abs(C_stack @ w)**2
            reshaped_powers = anp.reshape(power_terms,(self.nEl, dim), order='F')
            power_sum_vector = anp.sum(reshaped_powers, axis=1)

            # Soft Gain constraint 
            gain_value = -1 * anp.real(coeffs @ w) / self.nEl 
            return 30+ 10*anp.log10(anp.max(power_sum_vector)) + mu * gain_value
        
        problem = pymanopt.Problem(manifold=manifold, cost=cost)
        optimizer = pymanopt.optimizers.SteepestDescent(min_gradient_norm = 1e-9)

        result = optimizer.run(problem)

        return result.point

    def solver_FASTA(self, h5_filepath=None, dataset_name=None, s_data_cube=None, thresh=1e-9,
                    max_iter=10000, debug=True, device="cpu", method_type='RGD_momentum',
                    initial_tau=0.1, newton_tol=1e-5, newton_max_iter=50, momentum=0.9,
                    progress_every=None, progress_label=None):
        """
        Solves the optimization problem using FASTA-like methods with data from an HDF5 file
        or an in-memory tensor.

        Args:
            h5_filepath (str): Path to the HDF5 file.
            dataset_name (str): The name of the dataset within the HDF5 file.
            s_data_cube (torch.Tensor): In-memory data cube.
            thresh (float): Relative tolerance for convergence (used for logging).
            max_iter (int): Maximum number of iterations.
            debug (bool): If True, records objective values for analysis.
            device (str): The compute device ('cuda' or 'cpu').
            method_type (str): 'PGD', 'PGD_BB', 'RGD_no_momentum', 'RGD_momentum', or 'RGD_NewtonCG'.
            initial_tau (float): The starting step size (tau_0) for the diminishing step rule.
            newton_tol (float): Gradient norm tolerance for the Newton method.
            newton_max_iter (int): Max iterations for the TCG inner solve.
            momentum (float): Momentum factor for PGD (0.0 means standard PGD).
        """
        assert (h5_filepath is not None and dataset_name is not None) or (s_data_cube is not None), \
            "Must provide either HDF5 path and dataset name, or an in-memory s_data_cube tensor."
        assert not ((h5_filepath is not None) and (s_data_cube is not None)), \
            "Cannot provide both HDF5 path and an in-memory tensor."
        
        self.device = torch.device(device)
        self.initial_tau = initial_tau 
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.momentum_beta = momentum

        # --- 1. Initialization ---
        if s_data_cube is not None:
            self.s_data_cube_tensor = s_data_cube.to(torch.complex64)
            # Place on GPU when it fits — _get_inf_norm_obj_value_ram_gpu's fast
            # path checks self.s_data_cube_conj.device. Without this the CPU
            # fallback einsum dominates per-iter cost.
            if self.device.type == 'cuda':
                data_bytes = self.s_data_cube_tensor.nelement() * self.s_data_cube_tensor.element_size()
                gpu_total = torch.cuda.get_device_properties(self.device).total_memory
                if data_bytes < gpu_total * 0.6:
                    self.s_data_cube_tensor = self.s_data_cube_tensor.to(self.device)
                else:
                    self.s_data_cube_tensor = self.s_data_cube_tensor.pin_memory()
                    self._gpu_batch_size = max(1, int(gpu_total * 0.25 / (self.nEl * self.nEl * 8)))
            self.s_data_cube_conj = self.s_data_cube_tensor.conj()
            self._objective_function = self._get_inf_norm_obj_value_ram_gpu
            self._get_slice = lambda idx: self.s_data_cube_tensor[idx].to(self.device)
        else:
            self.h5_filepath = h5_filepath
            self.dataset_name = dataset_name
            self._objective_function = self._get_inf_norm_obj_value_hdf5_gpu
            self._get_slice = self._get_slice_from_hdf5

        # Initialize weights to the conjugate beamforming weights a_c, matching
        # iMM's initialization (and what the paper's Sec V states). Previously
        # this was torch.ones(...) which is broadside, NOT conjugate, and caused
        # PsGM/RsGM curves to start at a different value than iMM.
        w = torch.asarray(self.a_c).to(torch.complex64).to(self.device)
        
        # Initialize states for momentum methods
        self.w_prev = torch.zeros_like(w) # For BB methods (position)
        self.v_prev = torch.zeros_like(w) # For Momentum methods (velocity vector)
        self.t_prev = 1.0
        
        self.a_tilda_torch = torch.from_numpy(self.a_tilda).to(torch.complex64).to(self.device)
        self.constraint_normal = torch.conj(self.a_tilda_torch)

        if "BB" in method_type:
            phase = torch.rand(size=(self.nEl, )).to(torch.complex64).to(device)
            self.w_prev = torch.exp(1j * phase)
            self.prev_obj_value,_ = self._objective_function(self.w_prev)
            self.t_min = 1e-5
            self.t_max = 50
            
        if debug:
            self.record_obj_value = []
            self.dual_value = []
            self.primal_value = []
            self.mu = [] 
        
        prev_obj_value, _ = self._objective_function(w)

        if debug:
            # _objective_function returns max_val/(Z_0*nEl); recover max_val by
            # multiplying back, then divide by total_input_power = Sum_k|X|^2
            # to land in true dB-Gamma^2.
            self.record_obj_value.append(10 * np.log10(prev_obj_value * 50 * self.nEl / self.total_input_power))

        # --- 2. Main Iteration Loop ---
        _label = progress_label or method_type
        _t0 = time.time()
        for iter_num in range(1, int(max_iter) + 1):

            # Call the unified update function
            if "BB" in method_type:
                self.t_max = 50 / np.log10(iter_num + 1)

            w, grad_norm = self._update_weights(w, iter_num, method_type)

            # For Newton method, check its specific convergence
            if method_type == 'RGD_NewtonCG' and grad_norm < self.newton_tol and iter_num > 10:
                print(f"Newton method converged at iteration {iter_num} (grad norm < {self.newton_tol}).")
                break

            current_obj_value, _ = self._objective_function(w)

            if debug:
                self.record_obj_value.append(10 * np.log10(current_obj_value * 50 * self.nEl / self.total_input_power))

            if progress_every and (iter_num % progress_every == 0 or iter_num == 1):
                _elapsed = time.time() - _t0
                _rate = iter_num / _elapsed if _elapsed > 0 else 0.0
                _remain = (int(max_iter) - iter_num) / _rate if _rate > 0 else float('inf')
                _last = self.record_obj_value[-1] if (debug and self.record_obj_value) else float('nan')
                print(f"[{_label}] iter {iter_num}/{int(max_iter)}  "
                      f"obj={_last:+.4f} dB  rate={_rate:.2f} it/s  "
                      f"elapsed={_elapsed:.0f}s  eta={_remain:.0f}s",
                      flush=True)
            
            # --- 3. Check for Convergence and Divergence ---
            if current_obj_value > prev_obj_value:
                relative_increase = (current_obj_value - prev_obj_value) / abs(prev_obj_value)
                if relative_increase > 0.1 and iter_num > 50: # Allow some initial fluctuation
                    print(f"Warning: Objective increased significantly at iteration {iter_num}. Stopping.")
                    break
            
            prev_obj_value = current_obj_value
        else:
            print(f"Reached maximum iterations ({max_iter}).")

        return w, getattr(self, 'record_obj_value', []), \
               getattr(self, 'dual_value', []), getattr(self, 'primal_value', []), \
               getattr(self, 'mu', [])

    def _get_inf_norm_obj_value_ram_gpu(self, w):
        """Calculates objective value, leveraging GPU when available."""
        data_device = self.s_data_cube_conj.device
        num_slices = self.s_data_cube_conj.shape[0]

        with torch.no_grad():
            if data_device.type == 'cuda':
                # Data on GPU — single vectorized pass
                w_dev = w.to(data_device)
                Sw = torch.einsum('bjk,k->bj', self.s_data_cube_conj, w_dev)
                vec = torch.real(torch.einsum('j,bj->b', w_dev.conj(), Sw))
                max_val, max_idx = torch.max(vec, dim=0)
                return max_val.item() / (50.0 * self.nEl), max_idx.item()

            elif hasattr(self, '_gpu_batch_size') and self.device.type == 'cuda':
                # Data pinned on CPU — compute in GPU batches
                w_gpu = w.to(self.device)
                w_conj = w_gpu.conj()
                batch_size = self._gpu_batch_size
                global_max_val = float('-inf')
                global_max_idx = 0

                for i in range(0, num_slices, batch_size):
                    end = min(i + batch_size, num_slices)
                    S_batch = self.s_data_cube_conj[i:end].to(self.device, non_blocking=True)
                    vec = torch.real(torch.einsum('j,bjk,k->b', w_conj, S_batch, w_gpu))
                    batch_max, batch_idx = torch.max(vec, dim=0)
                    bval = batch_max.item()
                    if bval > global_max_val:
                        global_max_val = bval
                        global_max_idx = i + batch_idx.item()

                return global_max_val / (50.0 * self.nEl), global_max_idx

            else:
                # Pure CPU path
                w_cpu = w.cpu() if w.device.type != 'cpu' else w
                vec = torch.real(torch.einsum('j,bjk,k->b', w_cpu.conj(), self.s_data_cube_conj, w_cpu))
                max_val, max_idx = torch.max(vec, dim=0)
                return max_val.item() / (50.0 * self.nEl), max_idx.item()

    def _get_slice_from_hdf5(self, index):
        """Reads a single slice of the S-matrix from the HDF5 file."""
        with h5py.File(self.h5_filepath, 'r') as hf:
            s_slice_np = hf[self.dataset_name][index]
        return torch.from_numpy(s_slice_np.astype(np.complex64)).to(self.device)
    
    def _get_inf_norm_obj_value_hdf5_gpu(self, w, batch_size=256):
        """
        Calculates objective value from HDF5, processing in batches on the GPU.
        """
        global_max_val = torch.tensor(float('-inf'), device=self.device, dtype=torch.float32)
        global_max_idx = torch.tensor(-1, device=self.device, dtype=torch.long)

        with h5py.File(self.h5_filepath, 'r') as hf:
            dset = hf[self.dataset_name]
            num_slices = dset.shape[0]

            with torch.no_grad():
                for i in range(0, num_slices, batch_size):
                    end_idx = min(i + batch_size, num_slices)
                    S_batch_np = dset[i:end_idx] # Read from disk
                    S_batch_gpu = torch.from_numpy(S_batch_np.astype(np.complex64)).to(self.device)
                    
                    vec_batch = torch.real(torch.einsum('j,bjk,k->b', w.conj(), S_batch_gpu.conj(), w))
                    
                    batch_max_val, batch_max_idx_relative = torch.max(vec_batch, dim=0)
                    
                    if batch_max_val > global_max_val:
                        global_max_val = batch_max_val
                        global_max_idx = i + batch_max_idx_relative

        return global_max_val.item() / (50.0 * self.nEl), global_max_idx.item()

    def _get_riemannian_gradient(self, w_curr, eu_grad):
        """
        Projects the Euclidean gradient onto the tangent cone of the manifold.
        """
        g_circle = eu_grad - torch.real(eu_grad * w_curr.conj()) * w_curr

        current_gain = self.evaluate_gain(w_curr)
        dist_to_boundary = current_gain - self.gmin
        
        is_active = dist_to_boundary < 1e-6
        
        if is_active:
            gain_derivative = torch.real(self.a_tilda_torch @ g_circle) 
            if gain_derivative > 0:
                n_vec = self.constraint_normal
                n_proj_circle = n_vec - torch.real(n_vec * w_curr.conj()) * w_curr
                
                norm_n_sq = torch.real(n_proj_circle.conj() @ n_proj_circle)
                
                if norm_n_sq > 1e-12:
                    projection_coeff = gain_derivative / norm_n_sq
                    g_riem = g_circle - projection_coeff * n_proj_circle
                    return g_riem

        return g_circle

    def _update_weights(self, w_curr, iter_num, method_type):
        """
        Performs a single update step for one of the specified methods.
        w_curr is w_k. self.w_prev is w_{k-1}.
        Returns:
            w_next (torch.Tensor): The updated weight vector.
            grad_norm (float): The norm of the gradient (0 for non-Newton methods).
        """
        
        # --- 1. Common Computations ---
        tau = self.initial_tau / (iter_num + 1)**0.5
        
        curr_obj, max_idx = self._objective_function(w_curr)
        S_slice = self._get_slice(max_idx)
        eu_grad_vec = 2 * (S_slice.conj() @ w_curr) / 50.0

        w_next = None
        grad_norm = 0.0 
        
        # --- 2. Method-Specific Update ---
        if method_type == 'PGD':
            # Projected Gradient Descent
            
            if self.momentum_beta > 0.0:
                
                self.v_prev = self.momentum_beta * self.v_prev + (1 - self.momentum_beta) * (-1 * eu_grad_vec)
                
                # Apply update
                w_intermediate = w_curr + tau * self.v_prev
                
            else:
                # Standard PGD (No Momentum)
                # w_{k+1} = w_k - tau * grad
                w_intermediate = w_curr - tau * eu_grad_vec

            # Project to feasible set
            w_next = self._project_to_feasible_set(w_intermediate)
            
           
        
        elif method_type == 'RGD_no_momentum':
            # 2) Riemannian Sub-gradient Descent (no momentum)
            r_grad_vec = self._get_riemannian_gradient(w_curr, eu_grad_vec)
            w_intermediate = w_curr - tau * r_grad_vec
            w_next = self._project_to_feasible_set(w_intermediate)
            
            self.t_prev = 1.0

        elif method_type == 'RGD_momentum':
            # 3) Riemannian Sub-gradient Descent with Momentum
            
            r_grad_vec = self._get_riemannian_gradient(w_curr, eu_grad_vec)
            v_transported = self.v_prev - torch.real(self.v_prev * w_curr.conj()) * w_curr
            v_new = self.momentum_beta * v_transported + (1 - self.momentum_beta) * r_grad_vec
            
            w_intermediate = w_curr - tau * v_new
            w_next = self._project_to_feasible_set(w_intermediate)

            self.v_prev = v_new.clone()
        
        elif "BB" in method_type:
            # Barzilai-Borwein methods
            delta_w = w_curr - self.w_prev
            delta_obj = self.prev_obj_value - curr_obj 
            
            denom = delta_obj + torch.real(eu_grad_vec.conj().T @ delta_w) if "RGD" in method_type else \
                    EPS + delta_obj + torch.real(eu_grad_vec.conj().T @ delta_w)
            
            tau_bb = 0.5 * torch.linalg.norm(delta_w)**2 / denom
            tau_bb = torch.clip(tau_bb, min=self.t_min, max=self.t_max)
            
            if method_type == 'PGD_BB':
                step_dir = eu_grad_vec
            else:
                step_dir = self._get_riemannian_gradient(w_curr, eu_grad_vec)
            
            w_intermediate = w_curr - tau_bb * step_dir
            w_next = self._project_to_feasible_set(w_intermediate)
            
            self.w_prev = w_curr.clone()
            self.prev_obj_value = curr_obj
            
        elif method_type == 'RGD_NewtonCG':
            objective_k = lambda w: torch.real(w.conj() @ S_slice.conj() @ w) / 50.0
            r_grad_vec = self._get_riemannian_gradient(w_curr, eu_grad_vec)
            grad_norm = torch.norm(r_grad_vec).item()
            hess_op = self._get_riemannian_hessian_op(w_curr, S_slice, objective_k)
            eta = self._tcg_solver(hess_op, -r_grad_vec, w_curr, max_iter=self.newton_max_iter)
            t = self._line_search(w_curr, eta, r_grad_vec, objective_k)
            w_intermediate = w_curr + t * eta
            w_next = self._project_to_feasible_set(w_intermediate)
            self.t_prev = 1.0
            self.w_prev = torch.zeros_like(w_curr)
            
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        return w_next,grad_norm
    
    def _get_riemannian_hessian_op(self, w, S_slice, objective_k):
        """
        Returns a function that computes the Hessian-vector product
        for the local smooth objective f_k(w) = w^H S_k w.
        Hess(w)[xi] = Proj( 2*S_k_conj*xi ) - 2*f_k(w)*xi
        """
        # Pre-calculate f_k(w)
        f_val_k = objective_k(w)
        # S_slice_conj is the matrix S_k in our formula f_k(w) = w^H S_k w
        S_slice_conj = S_slice.conj() / 50.0

        def hess_op(xi):
            # 1. Euclidean Hessian-vector product
            eu_hess_v = 2 * (S_slice_conj @ xi)
            
            # 2. Project Euclidean Hessian onto tangent space
            # Proj(v) = v - Re(v * w.conj()) * w
            proj_hess = eu_hess_v - torch.real(eu_hess_v * w.conj()) * w
            
            # 3. Add curvature term
            curvature_term = 2 * f_val_k * xi
            
            return proj_hess - curvature_term
            
        return hess_op

    def _tcg_solver(self, hess_op, b, w, max_iter=100, tol=1e-5):
        """
        Truncated Conjugate Gradient (TCG) solver in PyTorch.
        Solves Hess(eta) = b
        """
        eta = torch.zeros_like(w)
        r = b.clone()
        d = r.clone()
        
        # Use torch.real() for norms
        r_norm_sq = torch.real(r.conj() @ r)
        
        for _ in range(max_iter):
            if torch.sqrt(r_norm_sq) < tol:
                break # Solved
                
            Hess_d = hess_op(d)
            
            # We must use torch.real() for the dot product
            d_H_Hess_d = torch.real(d.conj() @ Hess_d)
            
            if d_H_Hess_d <= 0:
                # Negative curvature detected!
                # Return the current direction 'd' as it's a valid descent direction.
                return d
                
            alpha = r_norm_sq / d_H_Hess_d
            
            eta = eta + alpha * d
            r_new = r - alpha * Hess_d
            
            r_new_norm_sq = torch.real(r_new.conj() @ r_new)
            
            beta = r_new_norm_sq / r_norm_sq
            
            d = r_new + beta * d
            r = r_new
            r_norm_sq = r_new_norm_sq
            
        return eta

    def _line_search(self, w, eta, grad, objective_k, c=1e-4, tau=0.5, max_iter=20):
        """
        Armijo backtracking line search on the manifold.
        """
        t = 1.0  # Start with full Newton step
        f_w = objective_k(w) # Current value
        
        # Pre-calculate the slope
        slope = torch.real(grad.conj() @ eta)
        
        if slope >= 0:
            # This shouldn't happen with a proper TCG,
            # but as a safeguard, return a tiny step.
            return 1e-6 

        for _ in range(max_iter):
            # Test a new point by retracting the step
            w_new = self._project_to_feasible_set(w + t * eta)
            
            # Get the new function value
            f_new = objective_k(w_new)
            
            # Armijo condition: f(w_new) <= f(w) + c * t * slope
            if f_new <= f_w + c * t * slope:
                return t
            
            t = t * tau # Backtrack
            
        return t # Return last failed step size if loop finishes

    def _project_to_feasible_set(self, w_target, tol=1e-9):
        """
        Projects a vector w_target onto the intersection of the complex circle
        and the half-space defined by the gain constraint.
        """
        a_c_eff = torch.conj(self.a_tilda_torch)
        
        # Nested function to check gain. It captures 'self', 'w_target', 
        # and 'a_c_eff' from the enclosing scope.
        def check_gain(lam):
            w_candidate = torch.exp(1j * torch.angle(w_target + lam * a_c_eff))
            return self.evaluate_gain(w_candidate)

        # Check if the simple projection already works
        w_simple_proj = torch.exp(1j * torch.angle(w_target))
        if self.evaluate_gain(w_simple_proj) >= self.gmin:
            return w_simple_proj

        # If not, find the lambda that satisfies the constraint
        lambda_low, lambda_high = 0.0, 1e8 # Search range for lambda
        
        # Binary search for lambda
        for _ in range(100): # Max 100 iterations for binary search
            if (lambda_high - lambda_low) < tol:
                break
            lambda_mid = (lambda_low + lambda_high) / 2.0
            if check_gain(lambda_mid) < self.gmin:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
        
        lambda_opt = (lambda_low + lambda_high) / 2.0
        
        return torch.exp(1j * torch.angle(w_target + lambda_opt * a_c_eff))
    
    def evaluate_gain(self,w):
        return torch.real(self.a_tilda_torch @ w)