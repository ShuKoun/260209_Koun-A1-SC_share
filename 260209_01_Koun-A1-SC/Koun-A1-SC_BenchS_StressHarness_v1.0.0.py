"""
文件名 (Filename): BenchS_StressHarness_v1.4.6.py
中文標題 (Chinese Title): [Benchmark S] 壓力測試離心機 v1.4.6 (緩存鎖定終極版)
英文標題 (English Title): [Benchmark S] Stress Test Harness v1.4.6 (Final Cache Lock)
版本號 (Version): Harness v1.4.6
前置版本 (Prev Version): Harness v1.4.5 / Core v1.7.12

變更日誌 (Changelog):
    1. [Ops] 緩存鎖定：永久移除 main 循環末尾的 jax.clear_caches()，修正 v1.4.5 的緩存自殺回歸，確保 Warmup 編譯成果在全測試週期內有效。
    2. [Invariant] 黃金標準：保留 v1.4.5 的所有高級特性 (Bias-Triggered Probe, Norm-Controlled RHS, K-Space Schedule, Taxonomy)。
    3. [Meta] 封版：此版本為最終數據採集版本，不再接受任何非崩潰級別的修改。
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, jvp
from jax.scipy.sparse.linalg import gmres
import numpy as np
import time
from functools import partial
import pandas as pd
import sys
import gc

# 強制 64 位精度
jax.config.update("jax_enable_x64", True)

# ============================================================================
# 0. Configuration & Stress Parameters
# ============================================================================
q = 1.602e-19; kb = 1.38e-23; T = 300.0; eps_0 = 8.85e-14
Vt = (kb * T) / q
ni = 1.0e10; ni_vac = 1.0e-20

Lx = 1.0e-5; Ly = 0.5e-5

# [Stress Axis 1] Grid Density
GRID_LIST = [
    {'Nx': 120, 'Ny': 60,  'Tag': 'Std'},   # Standard
    {'Nx': 180, 'Ny': 90,  'Tag': 'Dense'}  # 1.5x Density
]

# [Stress Axis 2] Baseline Step Size
BASELINE_STEP_LIST = [0.2, 0.4]

# Case Definition (Physics identical to v1.7.12)
SCAN_PARAMS = [
    # Case 2: Asymmetric
    {'CaseID': 'C2', 'SlotW_nm': 2.0, 'N_high': 3e20, 'N_low': 1e17, 'BiasMax': 8.0, 'Q_trap': 0.0, 'Alpha': 0.4, 'RelayBias': 7.0, 'A1_Step': 0.1},
    
    # Case 3: The Killer
    {'CaseID': 'C3', 'SlotW_nm': 2.0, 'N_high': 1e21, 'N_low': 1e17, 'BiasMax': 8.0, 'Q_trap': 1.0e18, 'Alpha': 0.2, 'RelayBias': 4.0, 'A1_Step': 0.05},
]

# [Ops] Adaptive Budgeting
MAX_STEP_TIME_FIRST = 60.0  # Cold start allowance
MAX_STEP_TIME_NORMAL = 30.0 # Fast fail for continuation

# [Algo] Coarse-to-Fine Constants
COARSE_STRIDE = 5 # Jump 5 grid points at a time
FINE_BUFFER = 5   # Last 5 steps are fine-grained

BASELINE_PARAMS = {
    'max_iter': 30, 'tol': 1e-4, 
    'gmres_tol': 1e-2, 'gmres_maxiter': 80, 'gmres_restart': 20
}

A1_PARAMS = {
    'gmres_tol': 1e-1, 'gmres_maxiter': 30, 'gmres_restart': 5,
    'dt_reset': False, 'max_outer_iter': 50,
    'dt_max': 10.0,         
    'dt_growth_cap': 2.0,   
    'dt_shrink_noise': 0.8  
}

# ============================================================================
# 1. Kernels (JIT) - Identical to v1.7.12
# ============================================================================
@jit
def harmonic_mean(e1, e2): return 2.0 * e1 * e2 / (e1 + e2 + 1e-300)

@partial(jit, static_argnums=(3, 4))
def reconstruct_phi(phi_in, bias_L, bias_R, nx, ny):
    phi = jnp.zeros((ny, nx))
    phi = phi.at[:, 0].set(bias_L)
    phi = phi.at[:, -1].set(bias_R)
    ny_in, nx_in = ny - 2, nx - 2
    phi = phi.at[1:-1, 1:-1].set(phi_in.reshape((ny_in, nx_in)))
    phi = phi.at[0, :].set(phi[1, :])
    phi = phi.at[-1, :].set(phi[-2, :])
    return phi

@partial(jit, static_argnums=(9, 10))
def internal_residual(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    p_l = phi[1:-1, :-2]; p_r = phi[1:-1, 2:]
    p_u = phi[:-2, 1:-1]; p_d = phi[2:, 1:-1]
    e_c = eps_map[1:-1, 1:-1]
    e_l = eps_map[1:-1, :-2]; e_r = eps_map[1:-1, 2:]
    e_u = eps_map[:-2, 1:-1]; e_d = eps_map[2:, 1:-1]
    
    flux_r = harmonic_mean(e_c, e_r) * (p_r - p_c) / dx
    flux_l = harmonic_mean(e_c, e_l) * (p_c - p_l) / dx
    flux_d = harmonic_mean(e_c, e_d) * (p_d - p_c) / dy
    flux_u = harmonic_mean(e_c, e_u) * (p_c - p_u) / dy
    
    div_flux = (flux_r - flux_l)/dx + (flux_d - flux_u)/dy
    
    rho_free = q * (ni_map[1:-1, 1:-1] * (jnp.exp(-p_c/Vt) - jnp.exp(p_c/Vt)) + N_dop[1:-1, 1:-1])
    rho_trap = q * Q_trap_map[1:-1, 1:-1]
    
    return (div_flux + rho_free + rho_trap).flatten()

@partial(jit, static_argnums=(9, 10))
def merit_loss(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    res = internal_residual(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny)
    return 0.5 * jnp.sum(res**2)

compute_grad_merit = jit(grad(merit_loss, argnums=0), static_argnums=(9, 10))

@partial(jit, static_argnums=(9, 10))
def get_diag_precond(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    e_c = eps_map[1:-1, 1:-1]
    diag_pois = -2.0*e_c/(dx**2) - 2.0*e_c/(dy**2)
    term = -(q / Vt) * ni_map[1:-1, 1:-1] * (jnp.exp(-p_c/Vt) + jnp.exp(p_c/Vt))
    return (diag_pois + term).flatten()

@partial(jit, static_argnums=(10, 11)) 
def matvec_op_baseline(v, phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    _, Jv = jvp(lambda p: internal_residual(p, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny), (phi_in,), (v,))
    return Jv

@partial(jit, static_argnums=(12, 13)) 
def matvec_op_a1(v, phi_in, dt_inv, M_inv, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    _, Jv = jvp(lambda p: internal_residual(p, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny), (phi_in,), (v,))
    Av = v * dt_inv - Jv 
    return M_inv * Av    

# ============================================================================
# 2. Solvers (Instrumented with dynamic budget)
# ============================================================================
class BaselineNewton:
    def __init__(self, params): self.params = params

    def solve_step(self, phi_init, bias_L, bias_R, physics_args, step_time_limit=30.0):
        phi = phi_init
        nx, ny = physics_args[-2], physics_args[-1]
        start_time = time.time()
        
        t0 = time.time()
        res = internal_residual(phi, bias_L, bias_R, *physics_args)
        norm_init = float(jnp.linalg.norm(res)) 
        norm = norm_init
        t_res = time.time() - t0
        t_lin = 0.0; t_ls = 0.0
        last_gmres_info = 0 
        
        g_tol, g_max, g_rst = self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart']
        
        for k in range(self.params['max_iter']):
            if time.time() - start_time > step_time_limit:
                return phi, False, norm_init, norm, 0.0, k, "TIMEOUT", last_gmres_info, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)

            if k > 0:
                t0 = time.time()
                res = internal_residual(phi, bias_L, bias_R, *physics_args)
                norm = float(jnp.linalg.norm(res))
                t_res += time.time() - t0
            
            if norm < self.params['tol']: 
                rel = (norm_init - norm)/(norm_init+1e-12)
                return phi, True, norm_init, norm, rel, k, "CONVERGED", last_gmres_info, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)
            
            t0 = time.time()
            A_op = partial(matvec_op_baseline, 
                           phi_in=phi, bias_L=bias_L, bias_R=bias_R, 
                           eps_map=physics_args[0], N_dop=physics_args[1], ni_map=physics_args[2],
                           Q_trap_map=physics_args[3], 
                           dx=physics_args[4], dy=physics_args[5], 
                           nx=nx, ny=ny)
            
            RHS = -res 
            try:
                d, info = gmres(A_op, RHS, tol=g_tol, maxiter=g_max, restart=g_rst)
                d.block_until_ready() 
                last_gmres_info = info
            except:
                return phi, False, norm_init, norm, 0.0, k, "GMRES_EXCEPT", -1, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)
            
            if info > 0:
                 t_lin += time.time() - t0
                 return phi, False, norm_init, norm, 0.0, k, f"GMRES_FAIL", info, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)
            
            t_lin += time.time() - t0
            
            t0 = time.time()
            alpha = 1.0; merit_old = 0.5 * norm**2; success_ls = False; phi_next = phi
            for i in range(8):
                phi_try = phi + alpha * d
                merit_new = float(merit_loss(phi_try, bias_L, bias_R, *physics_args))
                if merit_new < merit_old:
                    phi_next = phi_try; success_ls = True; break
                alpha *= 0.5
            t_ls += time.time() - t0
            
            if success_ls: phi = phi_next
            else: return phi, False, norm_init, norm, 0.0, k, "LS_FAIL", last_gmres_info, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)
                
        return phi, False, norm_init, norm, 0.0, self.params['max_iter'], "MAX_ITER", last_gmres_info, t_lin, t_res, t_ls, (g_tol, g_max, g_rst)

class KounA1Solver:
    def __init__(self, params): 
        self.params = params
        self.dt = 1e-4
        
    def solve_step(self, phi_init, bias_L, bias_R, prev_dt, physics_args, is_relay_hard=False, step_time_limit=30.0):
        nx, ny = physics_args[-2], physics_args[-1]
        
        if self.params['dt_reset']: 
            self.dt = 1e-4
        else: 
            if is_relay_hard:
                self.dt = max(prev_dt, 1e-4) 
            else:
                self.dt = max(prev_dt * 0.5, 1e-4)
        
        self.dt = min(self.dt, self.params['dt_max'])
            
        phi = phi_init
        
        cnt_step = 0; cnt_noise = 0; cnt_sniper = 0
        start_time = time.time()
        t_lin = 0.0; t_ls = 0.0; t_res = 0.0
        last_gmres_info = 0
        
        t0 = time.time()
        res = internal_residual(phi, bias_L, bias_R, *physics_args)
        norm_init = float(jnp.linalg.norm(res))
        norm = norm_init
        t_res += time.time() - t0
        
        a1_stats = {'step':0, 'noise':0, 'sniper':0}
        captured_g_params = (self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart'])

        for k in range(self.params['max_outer_iter']):
            if time.time() - start_time > step_time_limit:
                a1_stats['step']=cnt_step; a1_stats['noise']=cnt_noise; a1_stats['sniper']=cnt_sniper
                return phi, False, norm_init, norm, 0.0, k, "TIMEOUT", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params

            if k > 0:
                t0 = time.time()
                res = internal_residual(phi, bias_L, bias_R, *physics_args)
                norm = float(jnp.linalg.norm(res))
                t_res += time.time() - t0
            
            merit = 0.5 * norm**2
            
            if norm < 1e-4: 
                rel = (norm_init - norm)/(norm_init+1e-12)
                a1_stats['step']=cnt_step; a1_stats['noise']=cnt_noise; a1_stats['sniper']=cnt_sniper
                return phi, True, norm_init, norm, rel, k, f"CONV({cnt_step}/{cnt_noise}/{cnt_sniper})", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params
            
            t0 = time.time()
            dt_inv = 1.0 / self.dt
            diag_J = get_diag_precond(phi, bias_L, bias_R, *physics_args)
            M_diag = dt_inv - diag_J 
            M_inv = 1.0 / (M_diag + 1e-12) 
            
            A_op_bound = partial(matvec_op_a1, 
                                 phi_in=phi, dt_inv=dt_inv, M_inv=M_inv, 
                                 bias_L=bias_L, bias_R=bias_R, 
                                 eps_map=physics_args[0], N_dop=physics_args[1], ni_map=physics_args[2],
                                 Q_trap_map=physics_args[3], 
                                 dx=physics_args[4], dy=physics_args[5], 
                                 nx=nx, ny=ny)
            
            RHS = M_inv * res 
            
            if is_relay_hard and norm_init > 1e1:
                tol_g, max_g, rst_g = 1e-2, 80, 20 
            else:
                tol_g, max_g, rst_g = self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart']
            
            captured_g_params = (tol_g, max_g, rst_g)

            d, info = gmres(A_op_bound, RHS, tol=tol_g, maxiter=max_g, restart=rst_g)
            d.block_until_ready()
            last_gmres_info = info
            t_lin += time.time() - t0
            
            t0 = time.time()
            alpha = 1.0; status = "FAIL"; phi_next = phi
            merit_new = merit # Init
            
            for i in range(8):
                step = alpha * d
                if jnp.max(jnp.abs(step)) > 0.5: step *= (0.5 / jnp.max(jnp.abs(step)))
                phi_try = phi + step
                merit_new = float(merit_loss(phi_try, bias_L, bias_R, *physics_args))
                
                if merit_new <= merit * (1.0 - 1e-12): status = "STEP"; phi_next = phi_try; break
                if merit_new <= merit + 1e-6: status = "NOISE"; phi_next = phi_try; break
                alpha *= 0.5
            t_ls += time.time() - t0
            
            if status != "FAIL":
                phi = phi_next
                rel_improve = (merit - merit_new) / (merit + 1e-12)
                if status == "STEP":
                    cnt_step += 1
                    if rel_improve > 1e-1: factor = 1.5
                    elif rel_improve > 1e-2: factor = 1.2
                    else: factor = 1.0
                    factor = min(factor, self.params['dt_growth_cap'])
                    self.dt *= factor
                else: 
                    cnt_noise += 1
                    self.dt *= self.params['dt_shrink_noise'] 
                    if self.dt < 1e-4: self.dt = 1e-4
            else:
                self.dt *= 0.2
                if self.dt < 1e-9: 
                    a1_stats['step']=cnt_step; a1_stats['noise']=cnt_noise; a1_stats['sniper']=cnt_sniper
                    return phi, False, norm_init, norm, 0.0, k, "DT_COLLAPSE", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params
            
            self.dt = min(self.dt, self.params['dt_max'])

        a1_stats['step']=cnt_step; a1_stats['noise']=cnt_noise; a1_stats['sniper']=cnt_sniper
        return phi, False, norm_init, norm, 0.0, k, "MAX_ITER", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params

# ============================================================================
# 3. Execution (Harness Logic v1.4.6)
# ============================================================================
def setup_materials(X, Y, params):
    slot_w = params['SlotW_nm'] * 1e-7 
    slot_h = 30.0 * 1e-7 
    y_center = 0.7 * Ly 
    mask_vac = (jnp.abs(X - Lx/2) < slot_w/2) & (jnp.abs(Y - y_center) < slot_h/2)
    mask_vac = mask_vac.astype(jnp.float64)
    mask_si = 1.0 - mask_vac
    eps_map = (mask_si * 11.7 + mask_vac * 1.0) * eps_0
    
    N_dop = (jnp.where(X < Lx/2, params['N_high'], params['N_low']) * mask_si)
    
    ni_map = mask_si * ni + mask_vac * ni_vac
    Q_trap_vol = params.get('Q_trap', 0.0)
    Q_trap_map = mask_vac * Q_trap_vol
    
    return eps_map, N_dop, ni_map, Q_trap_map

def setup_grid(nx, ny):
    dx = Lx / (nx - 1); dy = Ly / (ny - 1)
    x = jnp.linspace(0, Lx, nx); y = jnp.linspace(0, Ly, ny)
    X, Y = jnp.meshgrid(x, y)
    return X, Y, dx, dy

# [Ops v1.4.6] Cache-Safe Probe: Bias-Triggered + Norm-Controlled
def warmup_kernels():
    print("\n>>> JIT WARMUP: Compiling Isomorphic Operators for all Grids & Cases...")
    dt_inv = 1.0 / 1e-4
    
    for grid_cfg in GRID_LIST:
        nx, ny = grid_cfg['Nx'], grid_cfg['Ny']
        nx_i, ny_i = int(nx), int(ny)
        print(f"  [Warmup Grid] {grid_cfg['Tag']} ({nx_i}x{ny_i})...", end="")
        
        for p_idx, params in enumerate(SCAN_PARAMS):
            start_case = time.time()
            
            X, Y, dx, dy = setup_grid(nx_i, ny_i)
            eps_map, N_dop, ni_map, Q_trap_map = setup_materials(X, Y, params)
            
            phi_bc_L = Vt * jnp.log(params['N_high']/ni)
            phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni)
            alpha = params.get('Alpha', 0.0)
            phi_bc_R_base = phi_bc_L + alpha * (phi_bc_R_phys - phi_bc_L)
            
            x_lin = jnp.linspace(0, 1, nx_i)
            phi_row = phi_bc_L * (1.0 - x_lin) + phi_bc_R_phys * x_lin
            phi_full = jnp.tile(phi_row, (ny_i, 1))
            phi_init = phi_full[1:-1, 1:-1].flatten()
            
            relay_target = params['RelayBias']
            
            bias_mv_set = {0, int(round(relay_target * 1000))}
            for step_val in BASELINE_STEP_LIST:
                k = int(np.floor(relay_target / step_val + 1e-9))
                snapped_val = k * step_val
                bias_mv_set.add(int(round(snapped_val * 1000)))
            
            warmup_biases = sorted([mv / 1000.0 for mv in bias_mv_set])
            print(f" [Biases: {['{:.3f}'.format(b) for b in warmup_biases]}] ", end="")
            
            for w_bias in warmup_biases:
                bias_L = phi_bc_L
                bias_R = phi_bc_R_base + w_bias
                
                res = internal_residual(phi_init, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                res.block_until_ready()
                
                diag_J = get_diag_precond(phi_init, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                diag_J.block_until_ready()
                
                M_diag = dt_inv - diag_J
                M_inv = 1.0 / (M_diag + 1e-12)
                M_inv.block_until_ready()
                
                v_dummy = jnp.ones_like(phi_init)
                v_dummy.block_until_ready() 
                
                mv_b = matvec_op_baseline(v_dummy, phi_init, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                mv_b.block_until_ready()
                
                mv_a = matvec_op_a1(v_dummy, phi_init, dt_inv, M_inv, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                mv_a.block_until_ready()
                
                # [v1.4.5] Norm-Controlled Probe
                res_norm = jnp.linalg.norm(res) + 1e-300
                target_norm = 1e-3
                rhs_probe = res * (target_norm / res_norm)
                
                # Baseline GMRES
                A_op_base = partial(matvec_op_baseline, phi_in=phi_init, bias_L=bias_L, bias_R=bias_R, 
                                    eps_map=eps_map, N_dop=N_dop, ni_map=ni_map, Q_trap_map=Q_trap_map, 
                                    dx=dx, dy=dy, nx=nx_i, ny=ny_i)
                g_base = BASELINE_PARAMS
                x_b, _ = gmres(A_op_base, -rhs_probe, tol=g_base['gmres_tol'], maxiter=g_base['gmres_maxiter'], restart=g_base['gmres_restart']) 
                x_b.block_until_ready()
                
                # A1 Standard GMRES
                A_op_a1 = partial(matvec_op_a1, phi_in=phi_init, dt_inv=dt_inv, M_inv=M_inv, bias_L=bias_L, bias_R=bias_R, 
                                  eps_map=eps_map, N_dop=N_dop, ni_map=ni_map, Q_trap_map=Q_trap_map, 
                                  dx=dx, dy=dy, nx=nx_i, ny=ny_i)
                g_a1 = A1_PARAMS
                rhs_probe_a1 = M_inv * rhs_probe
                x_a1, _ = gmres(A_op_a1, rhs_probe_a1, tol=g_a1['gmres_tol'], maxiter=g_a1['gmres_maxiter'], restart=g_a1['gmres_restart']) 
                x_a1.block_until_ready()
                
                # A1 Hard GMRES
                x_h, _ = gmres(A_op_a1, rhs_probe_a1, tol=1e-2, maxiter=80, restart=20) 
                x_h.block_until_ready()
                
                # [v1.4.5] Bias-Triggered Hard Branch
                res_norm_val = float(jnp.linalg.norm(res))
                bias_R_probe = bias_R
                
                if res_norm_val <= 15.0: 
                    # Force Hard Relay Branch via Extreme Boundary Condition
                    bias_R_probe = phi_bc_R_base + params['BiasMax'] + 1.0
                
                physics_args = (eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                dummy_solver = KounA1Solver(A1_PARAMS)
                dummy_solver.solve_step(phi_init, bias_L, bias_R_probe, 1e-4, physics_args, is_relay_hard=True, step_time_limit=0.5)
            
            print(f" Done ({time.time()-start_case:.2f}s)")
    print(">>> JIT WARMUP COMPLETE.\n")

def run_sweep_stress(params, grid_cfg, base_step, solver_type, start_bias, stop_bias, init_phi=None, capture_k=None, relay_meta=None, a1_span=None):
    if solver_type == 'A1':
        assert a1_span is not None, "Error: A1 solver requires 'a1_span'."
    else:
        assert stop_bias is not None, "Error: Baseline solver requires 'stop_bias'."

    nx, ny = grid_cfg['Nx'], grid_cfg['Ny']
    
    step_val_log = base_step if solver_type == 'Baseline' else params['A1_Step']
    print(f"  > [{solver_type}] Grid={grid_cfg['Tag']}({nx}x{ny}), Step={step_val_log}V, Range={start_bias:.2f}->...", end="")
    
    X, Y, dx, dy = setup_grid(nx, ny)
    eps_map, N_dop, ni_map, Q_trap_map = setup_materials(X, Y, params)
    physics_args = (eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny)
    
    phi_bc_L = Vt * jnp.log(params['N_high']/ni)
    phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni)
    alpha = params.get('Alpha', 0.0)
    phi_bc_R_base = phi_bc_L + alpha * (phi_bc_R_phys - phi_bc_L)
    
    if init_phi is not None:
        phi_init = jnp.array(init_phi)
    else:
        # Ramp start
        phi_seed_L = phi_bc_L
        phi_seed_R = phi_bc_R_phys
        phi_full = jnp.zeros((ny, nx))
        for i in range(nx):
            r = i/(nx-1)
            phi_full = phi_full.at[:, i].set(phi_seed_L*(1-r) + phi_seed_R*r)
        phi_init = phi_full[1:-1, 1:-1].flatten()
        
    solver = KounA1Solver(A1_PARAMS) if solver_type == 'A1' else BaselineNewton(BASELINE_PARAMS)
    current_dt = 1e-4
    
    step_val = base_step if solver_type == 'Baseline' else params['A1_Step'] 
    
    # [Algo v1.3.2] K-Space Coarse-to-Fine Scheduling (Saturated)
    bias_points = []
    
    if solver_type == 'A1':
        k_start = int(np.round(start_bias / step_val))
        n_steps_sprint = int(np.round(a1_span / step_val))
        k_end = k_start + n_steps_sprint
        print(f" (Sprint: {n_steps_sprint} steps)")
        
        # [v1.3.2 Data] Anchor has step_exec=0.0
        bias_points.append((k_start, k_start*step_val, 0.0))
        for k in range(k_start + 1, k_end + 1):
            bias_points.append((k, k * step_val, step_val))
            
    else:
        # Baseline: K-Space Coarse-to-Fine
        if capture_k is not None:
            k_target = capture_k
        else:
            k_target = int(np.floor(stop_bias / step_val + 1e-9))
            
        k_switch = max(0, k_target - FINE_BUFFER)
        
        # Coarse Phase
        k_curr = 0
        bias_points.append((k_curr, k_curr * step_val, 0.0)) # [v1.3.2] Anchor=0.0
        
        while k_curr < k_switch:
            # [v1.3.2 Algo] Saturated stride (min)
            k_next = min(k_curr + COARSE_STRIDE, k_switch)
            if k_next == k_curr: break
            
            step_size_eff = (k_next - k_curr) * step_val
            k_curr = k_next
            bias_points.append((k_curr, k_curr * step_val, step_size_eff))
            
        # Fine Phase
        while k_curr < k_target:
            k_curr += 1
            bias_points.append((k_curr, k_curr * step_val, step_val))
            
        print(f"{k_target * step_val:.2f}V (K-Space C2F)")

    
    results = []
    last_phi = phi_init
    success_max_bias = np.nan 
    fail_reason = "NONE"
    captured_phi = None 
    n_steps_sprint = 0 
    if solver_type == 'A1': n_steps_sprint = int(np.round(a1_span / step_val))

    is_relay_run = (solver_type == 'A1' and init_phi is not None)
    relay_dt_boost_done = False
    
    # Run the schedule
    for step_idx, (k_idx, bias_val, current_step_size) in enumerate(bias_points):
        bias = bias_val
        bc_R = phi_bc_R_base + bias
        
        start = time.time()
        
        # [v1.3.2 Data] Budget logic by index
        is_first_step = (step_idx == 0)
        budget = MAX_STEP_TIME_FIRST if is_first_step else MAX_STEP_TIME_NORMAL
        
        if solver_type == 'A1':
             phi_next, succ, _, res1, rel, iters, extra, last_gmres, next_dt, t_lin, t_res, t_ls, a1_counts, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, current_dt, physics_args, is_relay_hard=is_relay_run, step_time_limit=budget)
             current_dt = next_dt
             if is_relay_run and (not relay_dt_boost_done) and iters == 0:
                current_dt = 1e-3; relay_dt_boost_done = True
        else:
             phi_next, succ, _, res1, rel, iters, extra, last_gmres, t_lin, t_res, t_ls, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, physics_args, step_time_limit=budget)
             a1_counts = {'step':0,'noise':0,'sniper':0} 
        
        dur = time.time() - start
        
        raw_status = extra.split('(')[0] if '(' in extra else extra
        status_kind = 'CONV' if raw_status == 'CONVERGED' else raw_status
        is_converged = (status_kind == 'CONV')
        gmres_boosted = (g_params == (1e-2, 80, 20))
        
        # [v1.3.3 Data] Final Semantic Lock
        fail_class = "NONE"
        if not succ:
            if "TIMEOUT" in extra: fail_class = "BUDGET_TIMEOUT"
            elif any(s in extra for s in ["GMRES", "LS_FAIL", "DT", "MAX_ITER"]): fail_class = "NUMERIC_FAIL"
            else: fail_class = "OTHER"
        elif is_converged:
            fail_class = "CONV"
            
        is_anchor = (step_idx == 0 and solver_type == 'A1') # [v1.3.3] Explicit definition

        row = {
            'solver': solver_type,
            'case_id': params['CaseID'],
            'grid_tag': grid_cfg['Tag'],
            'base_step': base_step,
            'bias': bias,
            'status': extra,
            'status_kind': status_kind,
            'is_converged': is_converged,
            'fail_class': fail_class, 
            'iters': iters,
            'time': dur,
            'res1': res1,
            'gmres_boosted': gmres_boosted,
            'step_exec': current_step_size, # [v1.3.3] Renamed for clarity
            'is_anchor': is_anchor,
            'is_relay_mode': is_relay_run, # [v1.3.3] New column
            'k_idx': k_idx,
            'k_start': bias_points[0][0] if len(bias_points)>0 else 0, 
            'dt': float(current_dt) if solver_type == 'A1' else 0.0,
            't_lin': t_lin, 't_res': t_res, 't_ls': t_ls,
            'g_tol': g_params[0], 'g_max': g_params[1], 'g_rst': g_params[2],
            'step_budget': budget,
            'is_first_step': is_first_step
        }
        if relay_meta:
            row.update(relay_meta)
            
        results.append(row)
        
        if succ:
            last_phi = phi_next
            success_max_bias = bias
            if capture_k is not None and k_idx == capture_k:
                captured_phi = phi_next
        else:
            fail_reason = extra
            print(f"    !!! FAILED at {bias:.2f}V ({extra})")
            break
            
    return pd.DataFrame(results), last_phi, success_max_bias, fail_reason, captured_phi, n_steps_sprint

def main():
    # [Ops v1.4.6] Cache Integrity Lock: jax.clear_caches() REMOVED.
    # This ensures the Warmup compilation artifacts persist throughout the sweep.
    gc.collect()
    warmup_kernels()
    
    full_logs = []
    summary_logs = []
    
    print("=== BENCHMARK S: STRESS HARNESS v1.4.6 (FINAL CACHE LOCK) ===")
    print(f"Grid List: {[g['Tag'] for g in GRID_LIST]}")
    print(f"Step List: {BASELINE_STEP_LIST}")
    print(f"Time Budget: First={MAX_STEP_TIME_FIRST}s (Hot), Normal={MAX_STEP_TIME_NORMAL}s")
    
    for grid_cfg in GRID_LIST:
        for base_step in BASELINE_STEP_LIST:
            print(f"\n>>> Stress Group: {grid_cfg['Tag']} | Step={base_step}V")
            
            for params in SCAN_PARAMS:
                case_id = params['CaseID']
                
                # [Integrity] Strict Snapping
                relay_target = params['RelayBias']
                relay_k = int(np.floor(relay_target / base_step + 1e-9))
                relay_bias_snapped = relay_k * base_step
                snap_delta = relay_bias_snapped - relay_target
                
                relay_meta = {
                    'relay_target': relay_target,
                    'relay_snapped': relay_bias_snapped,
                    'relay_delta': snap_delta
                }
                
                print(f"  # RelaySnap: Target={relay_target}V -> {relay_bias_snapped:.2f}V (Step {base_step}V, Delta {snap_delta:.2e}V)")

                # 1. Baseline
                stop_bias_base = relay_bias_snapped 
                
                df_base, _, max_bias_base, fail_reason_base, phi_relay, _ = run_sweep_stress(
                    params, grid_cfg, base_step, 'Baseline', 
                    start_bias=0.0, stop_bias=stop_bias_base,
                    capture_k=relay_k, relay_meta=relay_meta
                )
                full_logs.append(df_base)
                
                # Determine generic fail class for summary
                base_fail_class = "NONE"
                if not df_base.empty:
                    last_row = df_base.iloc[-1]
                    if not last_row['is_converged']:
                        base_fail_class = last_row['fail_class']
                
                summary_logs.append({
                    'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                    'solver': 'Baseline', 'max_bias': max_bias_base, 'fail_reason': fail_reason_base,
                    'fail_class': base_fail_class, 
                    'total_time': df_base['time'].sum(),
                    'relay_target': relay_target, 'relay_snapped': relay_bias_snapped, 'relay_delta': snap_delta
                })
                
                # 2. A1 Relay (Run Short Span)
                base_succeeded = (phi_relay is not None) and (not np.isnan(max_bias_base)) and (max_bias_base >= relay_bias_snapped)
                
                if base_succeeded:
                    start_bias_a1 = relay_bias_snapped
                    
                    df_a1, _, max_bias_a1, fail_reason_a1, _, sprint_n = run_sweep_stress(
                        params, grid_cfg, base_step, 'A1',
                        start_bias=start_bias_a1, stop_bias=None, 
                        init_phi=phi_relay, relay_meta=relay_meta,
                        a1_span=0.5 
                    )
                    full_logs.append(df_a1)
                    
                    a1_fail_class = "NONE"
                    if not df_a1.empty:
                        last_row = df_a1.iloc[-1]
                        if not last_row['is_converged']:
                            a1_fail_class = last_row['fail_class']

                    summary_logs.append({
                        'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                        'solver': 'A1', 'max_bias': max_bias_a1, 'fail_reason': fail_reason_a1,
                        'fail_class': a1_fail_class,
                        'total_time': df_a1['time'].sum(),
                        'relay_target': relay_target, 'relay_snapped': relay_bias_snapped, 'relay_delta': snap_delta,
                        'sprint_n_steps': sprint_n 
                    })
                else:
                    reason = "CAPTURE_MISSING" if phi_relay is None else "EARLY_FAIL"
                    print(f"    [Skip A1] Baseline Issue: {reason} (Max {max_bias_base}V vs Snap {relay_bias_snapped}V)")
                    summary_logs.append({
                        'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                        'solver': 'A1', 'max_bias': -1, 'fail_reason': reason, 'fail_class': 'SKIPPED', 'total_time': 0,
                        'relay_target': relay_target, 'relay_snapped': relay_bias_snapped, 'relay_delta': snap_delta
                    })
                
                gc.collect()
                # [Ops v1.4.6] Cache Integrity Lock: jax.clear_caches() REMOVED.

    # Save
    pd.concat(full_logs).to_csv("Stress_v1.4.6_FullLog.csv", index=False)
    pd.DataFrame(summary_logs).to_csv("Stress_v1.4.6_Summary.csv", index=False)
    print("\n=== STRESS TEST COMPLETE ===")
    print("Saved: Stress_v1.4.6_FullLog.csv, Stress_v1.4.6_Summary.csv")

if __name__ == "__main__":
    main()