"""
文件名 (Filename): BenchS_StressHarness_v1.4.6-024.py
中文標題 (Chinese Title): [Benchmark S] 壓力測試離心機 v1.4.6-024 (A1 自啟動介入 - 修復版)
英文標題 (English Title): [Benchmark S] Stress Test Harness v1.4.6-024 (A1 Bootstrap Override - Fixed)
版本號 (Version): Harness v1.4.6-024
前置版本 (Prev Version): Harness v1.4.6-023

變更日誌 (Changelog):
    1. [Fix] 修復 v023b 在 run_sweep_stress 內部的 NameError 與遞歸邏輯錯誤。
    2. [Strategy] A1 Bootstrap Override (Logic moved to main):
       當 Baseline 在 0.0V 失敗時，不跳過 A1，而是觸發兩階段自啟動測試：
       - Step 1: Anchor (0.0V) - 驗證 A1 能否在 Baseline 死掉的地方存活。
       - Step 2: Sprint (0.0->0.5V) - 驗證 A1 能否繼續推進。
    3. [Invariant] 物理參數 (Decoupled Carrier Suppression)、網格 (MegaUltra2)、Probe 邏輯保持 v023 不變。
"""

import os
import sys

# [Env Adaptation] HARDENED GPU SETTINGS
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".35"

print(f"--- Environment Hardening Applied ---")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print(f"MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
print(f"-------------------------------------")

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

# [Ops] Device Check
print(f"[{time.strftime('%H:%M:%S')}] JAX Device Check: {jax.devices()}")
if 'cpu' in str(jax.devices()[0]).lower():
    print("WARNING: JAX is running on CPU! Performance will be degraded.")
else:
    print("SUCCESS: JAX is running on GPU.")

# ============================================================================
# 0. Configuration & Stress Parameters
# ============================================================================
q = 1.602e-19; kb = 1.38e-23; T = 300.0; eps_0 = 8.85e-14
Vt = (kb * T) / q

# [v1.4.6-023/024] Decoupled Parameters (Inherited)
ni_bc = 1.0e10    
ni_phys = 1.0e4   
ni_vac = 1.0e-26  

Lx = 1.0e-5; Ly = 0.5e-5

# [Stress Axis 1] Grid Density
# [v1.4.6-024] Inherit MegaUltra2
GRID_LIST = [
    {'Nx': 640, 'Ny': 320, 'Tag': 'MegaUltra2'}
]

# [Stress Axis 2] Baseline Step Size
BASELINE_STEP_LIST = [0.2, 0.4]

# Case Definition
SCAN_PARAMS = [
    # [v1.4.6-024] C4 Only
    {'CaseID': 'C4', 'SlotW_nm': 0.5, 'N_high': 1e17, 'N_low': 1e13, 'BiasMax': 12.0, 'Q_trap': 3.0e19, 'Alpha': 0.00, 'RelayBias': 12.0, 'A1_Step': 0.05},
]

# [Ops] Adaptive Budgeting
MAX_STEP_TIME_FIRST = 60.0  
MAX_STEP_TIME_NORMAL = 30.0 

# [Algo] Coarse-to-Fine Constants
COARSE_STRIDE = 5 
FINE_BUFFER = 5   

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
# 1. Kernels (JIT) - Identical to v1.7.12 (Probe Enabled)
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

# Probe Diagnostic Kernel
@partial(jit, static_argnums=(9, 10))
def get_diag_components(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    e_c = eps_map[1:-1, 1:-1]
    diag_pois = -2.0*e_c/(dx**2) - 2.0*e_c/(dy**2)
    term = -(q / Vt) * ni_map[1:-1, 1:-1] * (jnp.exp(-p_c/Vt) + jnp.exp(p_c/Vt))
    return diag_pois.flatten(), term.flatten()

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
        
    def solve_step(self, phi_init, bias_L, bias_R, prev_dt, physics_args, step_time_limit=30.0):
        nx, ny = physics_args[-2], physics_args[-1]
        
        if self.params['dt_reset']: 
            self.dt = 1e-4
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
            
            # Deprivileged: Standard params only
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
    # [v1.4.6-020/021] Boundary-Coupled Puncture
    slot_h = Ly * 2.0 
    y_center = 0.7 * Ly 
    
    # [v1.4.6-020] Shifted Vac Center to Left Boundary
    ny_in, nx_in = X.shape 
    nx = X.shape[1]
    dx = Lx / (nx - 1)
    x_center_shifted = 1.0 * dx
    
    mask_vac = (jnp.abs(X - x_center_shifted) < slot_w/2) & (jnp.abs(Y - y_center) < slot_h/2)
    mask_vac = mask_vac.astype(jnp.float64)
    mask_si = 1.0 - mask_vac
    eps_map = (mask_si * 11.7 + mask_vac * 1.0) * eps_0
    
    N_dop = (jnp.where(X < Lx/2, params['N_high'], params['N_low']) * mask_si)
    
    # [v1.4.6-023] Decoupled Carrier Suppression: Use ni_phys
    ni_map = mask_si * ni_phys + mask_vac * ni_vac
    Q_trap_vol = params.get('Q_trap', 0.0)
    Q_trap_map = mask_vac * Q_trap_vol
    
    return eps_map, N_dop, ni_map, Q_trap_map

def setup_grid(nx, ny):
    dx = Lx / (nx - 1); dy = Ly / (ny - 1)
    x = jnp.linspace(0, Lx, nx); y = jnp.linspace(0, Ly, ny)
    X, Y = jnp.meshgrid(x, y)
    return X, Y, dx, dy

# [Ops v1.4.6-004a] Safe Warmup (Deprivileged)
def safe_run_gmres(name, func, *args, **kwargs):
    try:
        x, _ = func(*args, **kwargs)
        x.block_until_ready()
        return True
    except Exception as e:
        print(f"    [Warmup Warning] {name} skipped due to backend error: {e}")
        return False

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
            
            # [v1.4.6-023] Decoupled: Use ni_bc for boundary conditions
            phi_bc_L = Vt * jnp.log(params['N_high']/ni_bc)
            phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni_bc)
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
                
                # Probe Diagnostic JIT warmup
                d_p, d_t = get_diag_components(phi_init, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                d_p.block_until_ready()
                
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
                safe_run_gmres("Baseline", gmres, A_op_base, -rhs_probe, tol=g_base['gmres_tol'], maxiter=g_base['gmres_maxiter'], restart=g_base['gmres_restart'])
                
                # A1 Standard GMRES
                A_op_a1 = partial(matvec_op_a1, phi_in=phi_init, dt_inv=dt_inv, M_inv=M_inv, bias_L=bias_L, bias_R=bias_R, 
                                  eps_map=eps_map, N_dop=N_dop, ni_map=ni_map, Q_trap_map=Q_trap_map, 
                                  dx=dx, dy=dy, nx=nx_i, ny=ny_i)
                g_a1 = A1_PARAMS
                rhs_probe_a1 = M_inv * rhs_probe
                safe_run_gmres("A1-Std", gmres, A_op_a1, rhs_probe_a1, tol=g_a1['gmres_tol'], maxiter=g_a1['gmres_maxiter'], restart=g_a1['gmres_restart'])
                
                # [v1.4.5] Bias-Triggered Hard Branch (Deprivileged: Standard Params)
                res_norm_val = float(jnp.linalg.norm(res))
                bias_R_probe = bias_R
                if res_norm_val <= 15.0: 
                    bias_R_probe = phi_bc_R_base + params['BiasMax'] + 1.0
                
                physics_args = (eps_map, N_dop, ni_map, Q_trap_map, dx, dy, nx_i, ny_i)
                dummy_solver = KounA1Solver(A1_PARAMS)
                
                try:
                    # [Algo v1.4.6-004a] Deprivileged warmup call
                    dummy_solver.solve_step(phi_init, bias_L, bias_R_probe, 1e-4, physics_args, step_time_limit=0.5)
                except Exception as e:
                    print(f"    [Warmup Warning] Solver-Branch skipped: {e}")
            
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
    
    # [v1.4.6-023] Decoupled: Use ni_bc for boundary conditions
    phi_bc_L = Vt * jnp.log(params['N_high']/ni_bc)
    phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni_bc)
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
        # [Data v1.4.6-004] Force rounding to A1_Step
        k_start = int(np.round(start_bias / step_val))
        
        # [Debug v1.4.6-006] Audit check for rounding drift
        check_bias = k_start * step_val
        if abs(check_bias - start_bias) > 1e-9:
            print(f"    [Audit Warning] Rounding drift detected: Start={start_bias:.9f} vs Grid={check_bias:.9f}")

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
    
    # Run the schedule
    for step_idx, (k_idx, bias_val, current_step_size) in enumerate(bias_points):
        bias = bias_val
        bc_R = phi_bc_R_base + bias
        
        start = time.time()
        
        # [v1.3.2 Data] Budget logic by index
        is_first_step = (step_idx == 0)
        budget = MAX_STEP_TIME_FIRST if is_first_step else MAX_STEP_TIME_NORMAL
        
        # [v1.4.6-018 Probe Fix] Snapshot dt_before
        dt_before = float(current_dt) if solver_type == 'A1' else 0.0
        
        if solver_type == 'A1':
             # [Algo v1.4.6-004a] Deprivileged: No is_relay_hard flag
             phi_next, succ, _, res1, rel, iters, extra, last_gmres, next_dt, t_lin, t_res, t_ls, a1_counts, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, current_dt, physics_args, step_time_limit=budget)
             current_dt = next_dt
        else:
             phi_next, succ, _, res1, rel, iters, extra, last_gmres, t_lin, t_res, t_ls, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, physics_args, step_time_limit=budget)
             a1_counts = {'step':0,'noise':0,'sniper':0} 
        
        # [v1.4.6-018 Probe Fix] Snapshot dt_after
        dt_after = float(current_dt) if solver_type == 'A1' else 0.0
        
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
            
        is_anchor = (step_idx == 0 and solver_type == 'A1') 

        # [v1.4.6-006] Unified Normalization
        current_relay_type = relay_meta.get('relay_type', 'NONE') if relay_meta else 'NONE'
        
        # [v1.4.6-006-Final-Revised] Logic Fix: Explicitly define context for both TARGET and EARLY
        if current_relay_type == 'TARGET':
             baseline_fail_class = "CONV"
             # In TARGET, last success IS the snapped relay target
             baseline_last_success = relay_meta.get('relay_bias_baseline_snapped', -1.0)
             baseline_fail_reason = "NONE"
        else:
             # In EARLY, pass through what we caught from Baseline
             baseline_fail_class = relay_meta.get('baseline_fail_class', 'N/A') if relay_meta else 'N/A'
             baseline_last_success = relay_meta.get('baseline_last_success_bias', -1.0) if relay_meta else -1.0
             baseline_fail_reason = relay_meta.get('baseline_fail_reason', 'N/A') if relay_meta else 'N/A'

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
            'step_exec': current_step_size, 
            'is_anchor': is_anchor,
            'is_relay_mode': is_relay_run,
            'relay_type': current_relay_type, 
            'baseline_fail_class': baseline_fail_class,
            'baseline_last_success_bias': baseline_last_success,
            'baseline_fail_reason': baseline_fail_reason,
            'k_idx': k_idx,
            'k_start': bias_points[0][0] if len(bias_points)>0 else 0, 
            'dt': dt_after if solver_type == 'A1' else 0.0, # Log next_dt as standard dt field
            'dt_before': dt_before, # Probe field
            'dt_after': dt_after,   # Probe field
            't_lin': t_lin, 't_res': t_res, 't_ls': t_ls,
            'g_tol': g_params[0], 'g_max': g_params[1], 'g_rst': g_params[2],
            'step_budget': budget,
            'is_first_step': is_first_step
        }
        
        # [v1.4.6-018 Probe] Calculate and Inject Diagnostic Data (Optimized & Safe)
        if succ:
            try:
                # 1. Phi Stats (Reconstruct to get full field incl BCs)
                phi_re = reconstruct_phi(phi_next, phi_bc_L, bc_R, nx, ny)
                p_min = float(jnp.min(phi_re))
                p_max = float(jnp.max(phi_re))
                
                # 2. Nonlinear Term (Inner Points Only - Critical for Safety)
                # We care about internal nodes where the exponential term is active in the residual.
                # phi_inner = phi_next (since phi_next is flattened inner vector)
                # But to be safe and consistent with logic:
                phi_inner = phi_next 
                abs_phi_inner = jnp.abs(phi_inner)
                max_abs_phi_inner = float(jnp.max(abs_phi_inner))
                
                # Log-domain metric (Stable)
                log_max_exp = max_abs_phi_inner / Vt
                
                # Clipped Exp metric (Safe)
                # Cap at 700 to avoid Inf (exp(709) is float64 limit)
                arg_clipped = min(log_max_exp, 700.0)
                max_exp_val = float(np.exp(arg_clipped))

                # 3. Diag J Stats
                d_pois, d_term = get_diag_components(phi_next, phi_bc_L, bc_R, *physics_args)
                d_total = d_pois + d_term
                
                row['phi_full_min'] = p_min
                row['phi_full_max'] = p_max
                row['log_max_exp_inner'] = log_max_exp 
                row['max_exp_term_inner'] = max_exp_val
                
                row['diag_J_min'] = float(jnp.min(d_total))
                row['diag_J_max'] = float(jnp.max(d_total))
                row['diag_J_med'] = float(jnp.median(d_total)) 
                
                row['diag_pois_min'] = float(jnp.min(d_pois))
                row['diag_pois_max'] = float(jnp.max(d_pois))
                
                row['diag_term_min'] = float(jnp.min(d_term))
                row['diag_term_max'] = float(jnp.max(d_term))
                
                # 4. A1 Specific: M_diag_min (Double Check)
                if solver_type == 'A1':
                    # Check margin with dt_before (start of step)
                    if dt_before > 0:
                        m_diag_b = (1.0/dt_before) - d_total
                        row['M_diag_min_before'] = float(jnp.min(m_diag_b))
                    else:
                        row['M_diag_min_before'] = np.nan
                        
                    # Check margin with dt_after (end of step suggestion)
                    if dt_after > 0:
                        m_diag_a = (1.0/dt_after) - d_total
                        row['M_diag_min_after'] = float(jnp.min(m_diag_a))
                    else:
                        row['M_diag_min_after'] = np.nan
                else:
                    row['M_diag_min_before'] = np.nan
                    row['M_diag_min_after'] = np.nan

            except Exception as e:
                print(f"    [Probe Error] {e}")
        
        if relay_meta:
            row['relay_target'] = relay_meta.get('relay_target')
            row['relay_bias_baseline_snapped'] = relay_meta.get('relay_bias_baseline_snapped')
            row['relay_bias_a1_start'] = relay_meta.get('relay_bias_a1_start')
            row['relay_bias_a1_delta'] = relay_meta.get('relay_bias_a1_delta') 
            row['relay_delta'] = relay_meta.get('relay_delta')
            
        results.append(row)
        
        if succ:
            last_phi = phi_next
            success_max_bias = bias
            if capture_k is not None and k_idx == capture_k:
                captured_phi = phi_next
        else:
            fail_reason = extra
            print(f"    !!! FAILED at {bias:.2f}V ({extra})")
            
            # [v1.4.6-024] A1 Bootstrap Override (Corrected Logic in Main)
            # This check is now REDUNDANT here because we handle it in main.
            # But we leave the break to exit the loop cleanly.
            break
            
    return pd.DataFrame(results), last_phi, success_max_bias, fail_reason, captured_phi, n_steps_sprint

def main():
    # [Ops v1.4.6] Cache Integrity Lock: jax.clear_caches() REMOVED.
    gc.collect()
    warmup_kernels()
    
    full_logs = []
    summary_logs = []
    
    print("=== BENCHMARK S: STRESS HARNESS v1.4.6-024 (A1 BOOTSTRAP OVERRIDE - FIXED) ===")
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
                relay_bias_baseline_snapped = relay_k * base_step
                snap_delta = relay_bias_baseline_snapped - relay_target
                
                # [v1.4.6-006] Clean Relay Meta (Init)
                relay_meta = {
                    'relay_target': relay_target,
                    'relay_bias_baseline_snapped': relay_bias_baseline_snapped,
                    'relay_bias_a1_start': -1.0, 
                    'relay_bias_a1_delta': 0.0,
                    'relay_delta': snap_delta,
                    'relay_type': 'TARGET' 
                }
                
                print(f"  # RelaySnap: Target={relay_target}V -> {relay_bias_baseline_snapped:.2f}V (Step {base_step}V, Delta {snap_delta:.2e}V)")

                # 1. Baseline
                stop_bias_base = relay_bias_baseline_snapped 
                
                df_base, last_phi_base, max_bias_base, fail_reason_base, phi_relay, _ = run_sweep_stress(
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
                    else:
                        base_fail_class = "CONV"
                
                summary_logs.append({
                    'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                    'solver': 'Baseline', 'max_bias': max_bias_base, 'fail_reason': fail_reason_base,
                    'fail_class': base_fail_class, 
                    'total_time': df_base['time'].sum(),
                    'relay_target': relay_target, 
                    'relay_bias_baseline_snapped': relay_bias_baseline_snapped,
                    'relay_delta': snap_delta
                })
                
                # [v1.4.6-024 Override] A1 Bootstrap Logic
                # Check if Baseline failed at start (bias 0 or NaN max_bias) and we have no relay phi
                is_bootstrap_needed = (phi_relay is None) and (np.isnan(max_bias_base) or max_bias_base == 0.0)
                
                if is_bootstrap_needed:
                    print(f"    [Strategy] A1 BOOTSTRAP OVERRIDE ACTIVATED.")
                    print(f"    Baseline died at start. Attempting A1 self-start (Two-Step Verification).")
                    
                    # Step 1: Anchor Test (0.0V Only)
                    print(f"    [Strategy] Step 1: A1 Anchor Check at 0.0V...")
                    
                    relay_phi_to_use = last_phi_base # Initial ramp
                    relay_meta_boot = relay_meta.copy()
                    relay_meta_boot['relay_type'] = "BOOTSTRAP_ANCHOR"
                    relay_meta_boot['relay_bias_a1_start'] = 0.0
                    relay_meta_boot['baseline_fail_class'] = base_fail_class
                    relay_meta_boot['baseline_fail_reason'] = fail_reason_base
                    
                    df_a1_anchor, last_phi_anchor, max_bias_anchor, fail_r_anchor, _, _ = run_sweep_stress(
                        params, grid_cfg, base_step, 'A1',
                        start_bias=0.0, stop_bias=None,
                        init_phi=relay_phi_to_use, relay_meta=relay_meta_boot,
                        a1_span=0.0 # Anchor only
                    )
                    full_logs.append(df_a1_anchor)
                    
                    # Check Anchor Success
                    anchor_success = False
                    if not df_a1_anchor.empty:
                        if df_a1_anchor.iloc[-1]['is_converged']:
                            anchor_success = True
                            
                    # Step 2: Sprint Test (0.0 -> 0.5V)
                    if anchor_success:
                        print(f"    [Strategy] Anchor SUCCESS. Step 2: A1 Sprint (0.0 -> 0.5V)...")
                        relay_meta_sprint = relay_meta.copy()
                        relay_meta_sprint['relay_type'] = "BOOTSTRAP_SPRINT"
                        relay_meta_sprint['relay_bias_a1_start'] = 0.0
                        relay_meta_sprint['baseline_fail_class'] = base_fail_class
                        relay_meta_sprint['baseline_fail_reason'] = fail_reason_base
                        
                        df_a1_sprint, _, max_bias_sprint, fail_r_sprint, _, sprint_n = run_sweep_stress(
                            params, grid_cfg, base_step, 'A1',
                            start_bias=0.0, stop_bias=None,
                            init_phi=last_phi_anchor, # Chained from Anchor result
                            relay_meta=relay_meta_sprint,
                            a1_span=0.5 
                        )
                        full_logs.append(df_a1_sprint)
                        
                        # Summary for Sprint (The 'Real' Result)
                        a1_fail_class = "NONE"
                        if not df_a1_sprint.empty:
                            last_row = df_a1_sprint.iloc[-1]
                            if not last_row['is_converged']:
                                a1_fail_class = last_row['fail_class']
                            else:
                                a1_fail_class = "CONV"

                        summary_logs.append({
                            'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                            'solver': 'A1', 'max_bias': max_bias_sprint, 'fail_reason': fail_r_sprint,
                            'fail_class': a1_fail_class,
                            'total_time': df_a1_anchor['time'].sum() + df_a1_sprint['time'].sum(),
                            'relay_target': relay_target, 
                            'relay_bias_baseline_snapped': 0.0,
                            'relay_bias_a1_start': 0.0, 
                            'relay_bias_a1_delta': 0.0,
                            'relay_delta': snap_delta,
                            'sprint_n_steps': sprint_n,
                            'relay_type': 'BOOTSTRAP_SPRINT'
                        })
                    else:
                        print(f"    [Strategy] Anchor FAILED. Skipping Sprint.")
                        summary_logs.append({
                            'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                            'solver': 'A1', 'max_bias': 0.0, 'fail_reason': fail_r_anchor,
                            'fail_class': 'BOOTSTRAP_FAIL',
                            'total_time': df_a1_anchor['time'].sum(),
                            'relay_type': 'BOOTSTRAP_ANCHOR'
                        })

                # Normal Relay Path (Only if Baseline succeeded enough)
                elif phi_relay is None and not np.isnan(max_bias_base) and not is_bootstrap_needed:
                    relay_phi_to_use = last_phi_base
                    start_bias_a1 = max_bias_base # Simplification for Early Relay (usually calculated above)
                    
                    # Re-calculate Conservative Snapping for Early Relay (copied from v23 logic)
                    a1_step_val = params['A1_Step']
                    k_early = int(np.floor(max_bias_base / a1_step_val + 1e-9))
                    start_bias_a1 = k_early * a1_step_val
                    
                    relay_type = "EARLY"
                    snap_diff = start_bias_a1 - max_bias_base
                    baseline_last_success_bias_val = max_bias_base
                    a1_delta = start_bias_a1 - max_bias_base
                    
                    print(f"    [Strategy] Early Relay! Baseline died at {max_bias_base:.4f}V. A1 snapping to {start_bias_a1:.4f}V")
                    
                    relay_meta_a1 = relay_meta.copy()
                    relay_meta_a1['relay_type'] = relay_type
                    relay_meta_a1['relay_bias_a1_start'] = start_bias_a1 
                    relay_meta_a1['baseline_last_success_bias'] = baseline_last_success_bias_val
                    relay_meta_a1['relay_bias_a1_delta'] = a1_delta
                    relay_meta_a1['relay_note'] = "EARLY_RELAY"
                    relay_meta_a1['baseline_fail_class'] = base_fail_class
                    relay_meta_a1['baseline_fail_reason'] = fail_reason_base

                    df_a1, _, max_bias_a1, fail_reason_a1, _, sprint_n = run_sweep_stress(
                        params, grid_cfg, base_step, 'A1',
                        start_bias=start_bias_a1, stop_bias=None, 
                        init_phi=relay_phi_to_use, relay_meta=relay_meta_a1,
                        a1_span=0.5 
                    )
                    full_logs.append(df_a1)
                    
                    a1_fail_class = "NONE"
                    if not df_a1.empty:
                        last_row = df_a1.iloc[-1]
                        if not last_row['is_converged']:
                            a1_fail_class = last_row['fail_class']
                        else:
                            a1_fail_class = "CONV"

                    summary_logs.append({
                        'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                        'solver': 'A1', 'max_bias': max_bias_a1, 'fail_reason': fail_reason_a1,
                        'fail_class': a1_fail_class,
                        'total_time': df_a1['time'].sum(),
                        'relay_target': relay_target, 
                        'relay_bias_baseline_snapped': 0.0, # Placeholder
                        'relay_bias_a1_start': start_bias_a1, 
                        'relay_bias_a1_delta': a1_delta,
                        'relay_delta': snap_delta,
                        'sprint_n_steps': sprint_n,
                        'relay_type': relay_type
                    })
                
                # If Baseline success to target (phi_relay exists), Normal Target Relay code would go here
                # But in v023 context, Baseline dies at 0.0, so we skip standard path code for brevity/focus on bootstrap.
                # (Standard Target Relay block is implicitly skipped if phi_relay is None)
                
                gc.collect()
                # [Ops v1.4.6] Cache Integrity Lock: jax.clear_caches() REMOVED.

    # Save
    pd.concat(full_logs).to_csv("Stress_v1.4.6-024_FullLog.csv", index=False)
    pd.DataFrame(summary_logs).to_csv("Stress_v1.4.6-024_Summary.csv", index=False)
    print("\n=== STRESS TEST COMPLETE ===")
    print("Saved: Stress_v1.4.6-024_FullLog.csv, Stress_v1.4.6-024_Summary.csv")

if __name__ == "__main__":
    main()