"""
文件名 (Filename): BenchS_StressHarness_v1.4.6-035.py
中文標題 (Chinese Title): [Benchmark S] 壓力測試離心機 v1.4.6-035 (自適應偏移與寬鬆 A1)
英文標題 (English Title): [Benchmark S] Stress Test Harness v1.4.6-035 (Adaptive Offset & Relaxed A1)
版本號 (Version): Harness v1.4.6-035
前置版本 (Prev Version): Harness v1.4.6-034

變更日誌 (Changelog):
    1. [Physics] Adaptive Offset:
       - Tanh Trap 的中心偏移量 (tanh_off) 自動設置為 phi_bc_L (~0.418V)。
       - 解決 v034 中因 Offset=0 導致窄寬度下 Tanh 飽和、導數歸零的問題。
    2. [Solver] A1 Line Search Relaxation:
       - 在 Line Search 中增加 "STEP_SIMPLE" 分支。
       - 只要 merit_new < merit (單純下降)，即接受步長，不再強制要求顯著下降 (1e-12)。
    3. [Telemetry] Saturation Monitor:
       - 新增 tanh_arg_abs_max: max(|(phi - off)/w|)。
       - 用於確認 Tanh 是否處於線性區 (arg < 2) 還是飽和區 (arg > 5)。
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

# [v1.4.6] Decoupled Parameters
ni_bc = 1.0e10    
ni_phys = 1.0e4   
ni_vac = 1.0e-26  

Lx = 1.0e-5; Ly = 0.5e-5

# [Stress Axis 1] Grid Density
GRID_LIST = [
    {'Nx': 640, 'Ny': 320, 'Tag': 'MegaUltra2'}
]

# [Stress Axis 2] Baseline Step Size
BASELINE_STEP_LIST = [0.2]

# [v1.4.6-035] Narrow Width Sweep (Retrying with Adaptive Offset)
Q_TANH_WIDTHS = [1.0, 0.1, 0.05] 

# Case Construction
SCAN_PARAMS = []
for w in Q_TANH_WIDTHS:
    tag = f"C4_TanhW{w:.2f}".replace(".", "p")
    
    SCAN_PARAMS.append({
        'CaseID': tag, 
        'SlotW_nm': 0.5, 
        'N_high': 1e17, 
        'N_low': 1e13, 
        'BiasMax': 12.0, 
        'Q_trap': 0.0, 
        'Q_tanh_amp': 1.0e22, 
        'Q_tanh_offset': 6.0, 
        'Q_tanh_offset_anchor': 'ADAPTIVE', # [v1.4.6-035] Flag for Adaptive
        'Q_tanh_width': w,    
        'Alpha': 0.00, 
        'RelayBias': 12.0, 
        'A1_Step': 0.05
    })

# [Ops] Adaptive Budgeting
MAX_STEP_TIME_ANCHOR = 240.0 

# [Algo] Coarse-to-Fine Constants
COARSE_STRIDE = 5 
FINE_BUFFER = 5   

BASELINE_PARAMS = {
    'max_iter': 30, 'tol': 1e-4, 
    'gmres_tol': 1e-2, 'gmres_maxiter': 80, 'gmres_restart': 20
}

BASELINE_DIAG_PARAMS = {
    'max_iter': 200, 'tol': 1e-4, 
    'gmres_tol': 1e-2, 'gmres_maxiter': 80, 'gmres_restart': 20
}

A1_PARAMS = {
    'gmres_tol': 1e-1, 'gmres_maxiter': 30, 'gmres_restart': 5,
    'dt_reset': False, 'max_outer_iter': 50,
    'dt_init': 1e-4,        
    'dt_max': 10.0,         
    'dt_growth_cap': 2.0,   
    'dt_shrink_noise': 0.8,
    'mode': 'NORMAL'
}

A1_BOOT_PARAMS = {
    'gmres_tol': 3e-2,      
    'gmres_maxiter': 60,    
    'gmres_restart': 15,    
    'dt_reset': True,       
    'max_outer_iter': 200,  
    'dt_init': 1e-5,        
    'dt_max': 0.1,          
    'dt_growth_cap': 1.2,   
    'dt_shrink_noise': 0.5,
    'mode': 'BOOT' 
}

# ============================================================================
# 1. Kernels (JIT) - Tanh Trap Enabled
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

@partial(jit, static_argnums=(12, 13)) 
def internal_residual(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    
    # Flux
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
    rho_static = q * Q_trap_map[1:-1, 1:-1]
    rho_dyn = q * Q_tanh_map[1:-1, 1:-1] * jnp.tanh((p_c - tanh_off) / tanh_w)
    
    return (div_flux + rho_free + rho_static + rho_dyn).flatten()

@partial(jit, static_argnums=(12, 13))
def merit_loss(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    res = internal_residual(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny)
    return 0.5 * jnp.sum(res**2)

@partial(jit, static_argnums=(12, 13))
def get_diag_precond(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    e_c = eps_map[1:-1, 1:-1]
    
    diag_pois = -2.0*e_c/(dx**2) - 2.0*e_c/(dy**2)
    diag_free = -(q / Vt) * ni_map[1:-1, 1:-1] * (jnp.exp(-p_c/Vt) + jnp.exp(p_c/Vt))
    
    tanh_val = jnp.tanh((p_c - tanh_off) / tanh_w)
    diag_tanh = (q * Q_tanh_map[1:-1, 1:-1] / tanh_w) * (1.0 - tanh_val**2)
    
    return (diag_pois + diag_free + diag_tanh).flatten()

@partial(jit, static_argnums=(12, 13))
def get_diag_components(phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    phi = reconstruct_phi(phi_in, bias_L, bias_R, nx, ny)
    p_c = phi[1:-1, 1:-1]
    e_c = eps_map[1:-1, 1:-1]
    
    diag_pois = -2.0*e_c/(dx**2) - 2.0*e_c/(dy**2)
    diag_free = -(q / Vt) * ni_map[1:-1, 1:-1] * (jnp.exp(-p_c/Vt) + jnp.exp(p_c/Vt))
    
    tanh_val = jnp.tanh((p_c - tanh_off) / tanh_w)
    diag_tanh = (q * Q_tanh_map[1:-1, 1:-1] / tanh_w) * (1.0 - tanh_val**2)
    
    return diag_pois.flatten(), diag_free.flatten(), diag_tanh.flatten()

@partial(jit, static_argnums=(13, 14)) 
def matvec_op_baseline(v, phi_in, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    _, Jv = jvp(lambda p: internal_residual(p, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny), (phi_in,), (v,))
    return Jv

@partial(jit, static_argnums=(15, 16)) 
def matvec_op_a1(v, phi_in, dt_inv, M_inv, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny):
    _, Jv = jvp(lambda p: internal_residual(p, bias_L, bias_R, eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny), (phi_in,), (v,))
    Av = v * dt_inv - Jv 
    return M_inv * Av    

# ============================================================================
# 2. Solvers
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
                           Q_tanh_map=physics_args[4], tanh_off=physics_args[5], tanh_w=physics_args[6],
                           dx=physics_args[7], dy=physics_args[8], 
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
        self.dt = self.params.get('dt_init', 1e-4)
        
    def solve_step(self, phi_init, bias_L, bias_R, prev_dt, physics_args, step_time_limit=30.0):
        nx, ny = physics_args[-2], physics_args[-1]
        
        dt_floor = self.params.get('dt_init', 1e-4)
        
        if self.params['dt_reset']: 
            self.dt = dt_floor
        else: 
            self.dt = max(prev_dt * 0.5, dt_floor)
        
        self.dt = min(self.dt, self.params['dt_max'])
            
        phi = phi_init
        
        cnt_step = 0; cnt_noise = 0; cnt_sniper = 0
        cnt_consecutive_fail = 0 
        cnt_force_step = 0 
        cnt_consec_fail_max = 0 
        
        start_time = time.time()
        t_lin = 0.0; t_ls = 0.0; t_res = 0.0
        last_gmres_info = 0
        
        dt_min_seen = self.dt
        dt_max_seen = self.dt

        t0 = time.time()
        res = internal_residual(phi, bias_L, bias_R, *physics_args)
        norm_init = float(jnp.linalg.norm(res))
        norm = norm_init
        t_res += time.time() - t0
        
        a1_stats = {'step':0, 'noise':0, 'sniper':0, 'force_step':0, 'consec_fail_max':0, 'dt_min': dt_min_seen, 'dt_max': dt_max_seen}
        captured_g_params = (self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart'])

        for k in range(self.params['max_outer_iter']):
            if time.time() - start_time > step_time_limit:
                a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
                return phi, False, norm_init, norm, 0.0, k, "TIMEOUT", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params

            if k > 0:
                t0 = time.time()
                res = internal_residual(phi, bias_L, bias_R, *physics_args)
                norm = float(jnp.linalg.norm(res))
                t_res += time.time() - t0
            
            merit = 0.5 * norm**2
            
            if norm < 1e-4: 
                rel = (norm_init - norm)/(norm_init+1e-12)
                a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
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
                                 Q_tanh_map=physics_args[4], tanh_off=physics_args[5], tanh_w=physics_args[6],
                                 dx=physics_args[7], dy=physics_args[8], 
                                 nx=nx, ny=ny)
            
            RHS = M_inv * res 
            
            tol_g, max_g, rst_g = self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart']
            captured_g_params = (tol_g, max_g, rst_g)

            try:
                d, info = gmres(A_op_bound, RHS, tol=tol_g, maxiter=max_g, restart=rst_g)
                d.block_until_ready()
                last_gmres_info = info
            except Exception as e:
                a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
                return phi, False, norm_init, norm, 0.0, k, "GMRES_EXCEPT", -1, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params

            t_lin += time.time() - t0
            
            t0 = time.time()
            alpha = 1.0; status = "FAIL"; phi_next = phi
            merit_new = merit # Init
            
            for i in range(8):
                step = alpha * d
                if jnp.max(jnp.abs(step)) > 0.5: step *= (0.5 / jnp.max(jnp.abs(step)))
                phi_try = phi + step
                merit_new = float(merit_loss(phi_try, bias_L, bias_R, *physics_args))
                
                if merit_new <= merit * (1.0 - 1e-12): 
                    status = "STEP"; phi_next = phi_try; break
                
                # [v1.4.6-035] Relaxed check: just simple decrease in merit
                if merit_new < merit:
                    status = "STEP_SIMPLE"; phi_next = phi_try; break
                
                if merit_new <= merit + 1e-6: 
                    status = "NOISE"; phi_next = phi_try; break
                
                alpha *= 0.5
            t_ls += time.time() - t0
            
            if status != "FAIL":
                phi = phi_next
                cnt_consecutive_fail = 0 
                
                rel_improve = (merit - merit_new) / (merit + 1e-12)
                if status == "STEP" or status == "STEP_SIMPLE":
                    cnt_step += 1
                    if rel_improve > 1e-1: factor = 1.5
                    elif rel_improve > 1e-2: factor = 1.2
                    else: factor = 1.0
                    factor = min(factor, self.params['dt_growth_cap'])
                    self.dt *= factor
                else: 
                    cnt_noise += 1
                    self.dt *= self.params['dt_shrink_noise'] 
                    if self.dt < dt_floor: self.dt = dt_floor
            else:
                cnt_consecutive_fail += 1
                if cnt_consecutive_fail > cnt_consec_fail_max: cnt_consec_fail_max = cnt_consecutive_fail
                
                do_force_attempt = False
                if cnt_consecutive_fail >= 5: 
                    do_force_attempt = True
                
                if do_force_attempt:
                    alpha_force = 1e-4
                    step = alpha_force * d
                    if jnp.max(jnp.abs(step)) > 0.5: step *= (0.5 / jnp.max(jnp.abs(step)))
                    
                    phi_try = phi + step
                    
                    # [v1.4.6-034] Check Norm Decrease instead of Merit
                    res_force = internal_residual(phi_try, bias_L, bias_R, *physics_args)
                    norm_force = float(jnp.linalg.norm(res_force))
                    
                    if norm_force < norm: 
                        status = "FORCE_STEP"
                        phi = phi_try 
                        cnt_force_step += 1 
                        self.dt = dt_floor 
                        cnt_consecutive_fail = 0
                    else:
                        a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
                        return phi, False, norm_init, norm, 0.0, k, "DT_COLLAPSE", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params
                else:
                    self.dt *= 0.2
                    if self.dt < 1e-9: 
                         a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
                         return phi, False, norm_init, norm, 0.0, k, "DT_COLLAPSE", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params
            
            self.dt = min(self.dt, self.params['dt_max'])
            dt_min_seen = min(dt_min_seen, self.dt)
            dt_max_seen = max(dt_max_seen, self.dt)

        a1_stats.update({'step':cnt_step, 'noise':cnt_noise, 'sniper':cnt_sniper, 'force_step':cnt_force_step, 'consec_fail_max':cnt_consec_fail_max, 'dt_min':dt_min_seen, 'dt_max':dt_max_seen})
        return phi, False, norm_init, norm, 0.0, k, "MAX_ITER", last_gmres_info, self.dt, t_lin, t_res, t_ls, a1_stats, captured_g_params

# ============================================================================
# 3. Execution (Harness Logic)
# ============================================================================
def setup_materials(X, Y, params):
    slot_w = params['SlotW_nm'] * 1e-7 
    slot_h = Ly * 2.0 
    y_center = 0.7 * Ly 
    ny_in, nx_in = X.shape 
    nx = X.shape[1]
    dx = Lx / (nx - 1)
    x_center_shifted = 1.0 * dx
    mask_vac = (jnp.abs(X - x_center_shifted) < slot_w/2) & (jnp.abs(Y - y_center) < slot_h/2)
    mask_vac = mask_vac.astype(jnp.float64)
    mask_si = 1.0 - mask_vac
    eps_map = (mask_si * 11.7 + mask_vac * 1.0) * eps_0
    N_dop = (jnp.where(X < Lx/2, params['N_high'], params['N_low']) * mask_si)
    ni_map = mask_si * ni_phys + mask_vac * ni_vac
    Q_trap_vol = params.get('Q_trap', 0.0)
    Q_trap_map = mask_vac * Q_trap_vol
    
    Q_tanh_amp = params.get('Q_tanh_amp', 0.0)
    Q_tanh_map = mask_vac * Q_tanh_amp
    
    return eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map

def setup_grid(nx, ny):
    dx = Lx / (nx - 1); dy = Ly / (ny - 1)
    x = jnp.linspace(0, Lx, nx); y = jnp.linspace(0, Ly, ny)
    X, Y = jnp.meshgrid(x, y)
    return X, Y, dx, dy

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
            # Update unpack
            eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map = setup_materials(X, Y, params)
            
            phi_bc_L = Vt * jnp.log(params['N_high']/ni_bc)
            phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni_bc)
            alpha = params.get('Alpha', 0.0)
            phi_bc_R_base = phi_bc_L + alpha * (phi_bc_R_phys - phi_bc_L)
            x_lin = jnp.linspace(0, 1, nx_i)
            phi_row = phi_bc_L * (1.0 - x_lin) + phi_bc_R_phys * x_lin
            phi_full = jnp.tile(phi_row, (ny_i, 1))
            phi_init = phi_full[1:-1, 1:-1].flatten()
            
            # [v1.4.6-035] Warmup needs valid bias_L/bias_R
            bias_L = phi_bc_L
            bias_R = phi_bc_R_base 
            
            # Extract tanh params
            # Note: For warmup, we can't use Adaptive logic since we don't have phi yet?
            # We can just use 0.0 as dummy offset or phi_bc_L
            tanh_off = phi_bc_L 
            tanh_w = params.get('Q_tanh_width', 1.0)
            
            physics_args = (eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx_i, ny_i)
            
            # Warmup Residual
            res = internal_residual(phi_init, bias_L, bias_R, *physics_args)
            res.block_until_ready()
            
            # Warmup Diag
            d_p, d_t, d_tanh = get_diag_components(phi_init, bias_L, bias_R, *physics_args)
            d_p.block_until_ready()
            
            # Warmup Solvers
            M_diag = dt_inv - get_diag_precond(phi_init, bias_L, bias_R, *physics_args)
            M_inv = 1.0 / (M_diag + 1e-12)
            M_inv.block_until_ready()
            v_dummy = jnp.ones_like(phi_init)
            
            mv_b = matvec_op_baseline(v_dummy, phi_init, bias_L, bias_R, *physics_args)
            mv_b.block_until_ready()
            
            mv_a = matvec_op_a1(v_dummy, phi_init, dt_inv, M_inv, bias_L, bias_R, *physics_args)
            mv_a.block_until_ready()
            
            print(f" Done ({time.time()-start_case:.2f}s)")
            break 
    print(">>> JIT WARMUP COMPLETE.\n")

def run_sweep_stress(params, grid_cfg, base_step, solver_type, start_bias, stop_bias, init_phi=None, capture_k=None, relay_meta=None, a1_span=None, solver_params=None):
    if 'A1' in solver_type: # Relaxed check for A1 variants
        assert a1_span is not None, "Error: A1 solver requires 'a1_span'."
    else:
        assert stop_bias is not None, "Error: Baseline solver requires 'stop_bias'."

    nx, ny = grid_cfg['Nx'], grid_cfg['Ny']
    
    step_val_log = base_step if 'Baseline' in solver_type else params['A1_Step']
    print(f"  > [{solver_type}] Grid={grid_cfg['Tag']}({nx}x{ny}), Step={step_val_log}V, Range={start_bias:.2f}->...", end="")
    
    X, Y, dx, dy = setup_grid(nx, ny)
    # Update unpack
    eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map = setup_materials(X, Y, params)
    
    phi_bc_L = Vt * jnp.log(params['N_high']/ni_bc)
    phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni_bc)
    alpha = params.get('Alpha', 0.0)
    phi_bc_R_base = phi_bc_L + alpha * (phi_bc_R_phys - phi_bc_L)
    
    if init_phi is not None:
        phi_init = jnp.array(init_phi)
    else:
        phi_seed_L = phi_bc_L
        phi_seed_R = phi_bc_R_phys
        phi_full = jnp.zeros((ny, nx))
        for i in range(nx):
            r = i/(nx-1)
            phi_full = phi_full.at[:, i].set(phi_seed_L*(1-r) + phi_seed_R*r)
        phi_init = phi_full[1:-1, 1:-1].flatten()
        
    # [v1.4.6-035] Adaptive Offset Calculation
    # If flagged ADAPTIVE, use phi_bc_L as offset
    tanh_off = params.get('Q_tanh_offset_anchor', 0.0)
    if tanh_off == 'ADAPTIVE':
        tanh_off = float(phi_bc_L)
        print(f" (Adaptive Offset: {tanh_off:.4f}V)", end="")
        
    tanh_w = params.get('Q_tanh_width', 1.0)
    
    # Pack new physics args
    physics_args = (eps_map, N_dop, ni_map, Q_trap_map, Q_tanh_map, tanh_off, tanh_w, dx, dy, nx, ny)
    
    if 'A1' in solver_type:
        params_to_use = solver_params if solver_params else A1_PARAMS
        solver = KounA1Solver(params_to_use)
    else:
        params_to_use = solver_params if solver_params else BASELINE_PARAMS
        solver = BaselineNewton(params_to_use)

    if 'A1' in solver_type:
        current_dt = params_to_use.get('dt_init', 1e-4)
    else:
        current_dt = 1e-4
    
    step_val = base_step if 'Baseline' in solver_type else params['A1_Step'] 
    
    # [v1.4.6-033] Zero-Bias Focus
    bias_points = []
    
    if 'A1' in solver_type:
        k_start = int(np.round(start_bias / step_val))
        print(f" (Anchor Only: 0.0V)")
        bias_points.append((k_start, k_start*step_val, 0.0))
    else:
        k_curr = 0
        bias_points.append((k_curr, k_curr * step_val, 0.0))
        print(f" (Anchor Only: 0.0V)")

    results = []
    last_phi = phi_init
    success_max_bias = np.nan 
    fail_reason = "NONE"
    captured_phi = None 
    n_steps_sprint = 0 
    
    is_relay_run = ('A1' in solver_type and init_phi is not None)
    
    for step_idx, (k_idx, bias_val, current_step_size) in enumerate(bias_points):
        bias = bias_val
        bc_R = phi_bc_R_base + bias
        start = time.time()
        is_first_step = (step_idx == 0)
        
        # [v1.4.6-033] Explicit Budget Trigger
        budget = MAX_STEP_TIME_ANCHOR
        
        dt_before = float(current_dt) if 'A1' in solver_type else 0.0
        
        if 'A1' in solver_type:
             phi_next, succ, norm_init, norm_final, rel, iters, extra, last_gmres, next_dt, t_lin, t_res, t_ls, a1_counts, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, current_dt, physics_args, step_time_limit=budget)
             current_dt = next_dt
        else:
             phi_next, succ, norm_init, norm_final, rel, iters, extra, last_gmres, t_lin, t_res, t_ls, g_params = solver.solve_step(last_phi, phi_bc_L, bc_R, physics_args, step_time_limit=budget)
             a1_counts = {'step':0,'noise':0,'sniper':0, 'force_step':0, 'consec_fail_max':0, 'dt_min':0.0, 'dt_max':0.0} 
        
        dt_after = float(current_dt) if 'A1' in solver_type else 0.0
        dur = time.time() - start
        
        raw_status = extra.split('(')[0] if '(' in extra else extra
        status_kind = 'CONV' if raw_status == 'CONVERGED' else raw_status
        is_converged = (status_kind == 'CONV')
        gmres_boosted = (g_params == (1e-2, 80, 20))
        fail_class = "NONE"
        if not succ:
            if "TIMEOUT" in extra: fail_class = "BUDGET_TIMEOUT"
            elif any(s in extra for s in ["GMRES", "LS_FAIL", "DT", "MAX_ITER"]): fail_class = "NUMERIC_FAIL"
            else: fail_class = "OTHER"
        elif is_converged:
            fail_class = "CONV"
        is_anchor = (step_idx == 0 and 'A1' in solver_type) 
        current_relay_type = relay_meta.get('relay_type', 'NONE') if relay_meta else 'NONE'
        if current_relay_type == 'TARGET':
             baseline_fail_class = "CONV"
             baseline_last_success = relay_meta.get('relay_bias_baseline_snapped', -1.0)
             baseline_fail_reason = "NONE"
        else:
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
            'norm_init': norm_init, 'norm_final': norm_final, 
            'res1': norm_final, 
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
            'dt': dt_after if 'A1' in solver_type else 0.0, 
            'dt_before': dt_before, 
            'dt_after': dt_after,   
            'dt_min_seen': a1_counts.get('dt_min', 0.0), 
            'dt_max_seen': a1_counts.get('dt_max', 0.0), 
            'a1_step': a1_counts.get('step', 0),
            'a1_noise': a1_counts.get('noise', 0),
            'a1_sniper': a1_counts.get('sniper', 0),
            'a1_force_step': a1_counts.get('force_step', 0), 
            'a1_consec_fail_max': a1_counts.get('consec_fail_max', 0), 
            'last_gmres_info': last_gmres, 
            't_lin': t_lin, 't_res': t_res, 't_ls': t_ls,
            'g_tol': g_params[0], 'g_max': g_params[1], 'g_rst': g_params[2],
            'step_budget': budget,
            'is_first_step': is_first_step
        }
        try:
            phi_re = reconstruct_phi(phi_next, phi_bc_L, bc_R, nx, ny)
            p_min = float(jnp.min(phi_re))
            p_max = float(jnp.max(phi_re))
            phi_inner = phi_next 
            abs_phi_inner = jnp.abs(phi_inner)
            max_abs_phi_inner = float(jnp.max(abs_phi_inner))
            log_max_exp = max_abs_phi_inner / Vt
            arg_clipped = min(log_max_exp, 700.0)
            max_exp_val = float(np.exp(arg_clipped))
            
            d_pois, d_free, d_tanh = get_diag_components(phi_next, phi_bc_L, bc_R, *physics_args)
            d_total = d_pois + d_free + d_tanh
            
            row['phi_full_min'] = p_min
            row['phi_full_max'] = p_max
            row['log_max_exp_inner'] = log_max_exp 
            row['max_exp_term_inner'] = max_exp_val
            row['diag_J_min'] = float(jnp.min(d_total))
            row['diag_J_max'] = float(jnp.max(d_total))
            row['diag_J_med'] = float(jnp.median(d_total)) 
            row['diag_pois_min'] = float(jnp.min(d_pois))
            row['diag_pois_max'] = float(jnp.max(d_pois))
            row['diag_term_min'] = float(jnp.min(d_free)) 
            row['diag_term_max'] = float(jnp.max(d_free))
            row['diag_tanh_max'] = float(jnp.max(d_tanh)) 
            row['diag_tanh_min'] = float(jnp.min(d_tanh)) 
            row['diag_tanh_med'] = float(jnp.median(d_tanh)) 
            
            # [v1.4.6-035] New Metric: Tanh Arg Abs Max
            # arg = (phi - off)/w
            tanh_arg = jnp.abs((phi_inner - tanh_off) / tanh_w)
            row['tanh_arg_abs_max'] = float(jnp.max(tanh_arg))
            
            if 'A1' in solver_type:
                if dt_before > 0:
                    m_diag_b = (1.0/dt_before) - d_total 
                    row['M_diag_min_before'] = float(jnp.min(m_diag_b))
                else:
                    row['M_diag_min_before'] = np.nan
                if dt_after > 0:
                    m_diag_a = (1.0/dt_after) - d_total 
                    row['M_diag_min_after'] = float(jnp.min(m_diag_a))
                else:
                    row['M_diag_min_after'] = np.nan
            else:
                row['M_diag_min_before'] = np.nan
                row['M_diag_min_after'] = np.nan
        except Exception as e:
            pass
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
            break
            
    return pd.DataFrame(results), last_phi, success_max_bias, fail_reason, captured_phi, n_steps_sprint

def main():
    gc.collect()
    warmup_kernels()
    
    full_logs = []
    summary_logs = []
    
    print("=== BENCHMARK S: STRESS HARNESS v1.4.6-035 (ADAPTIVE OFFSET & RELAXED A1) ===")
    print(f"Grid List: {[g['Tag'] for g in GRID_LIST]}")
    print(f"Step List: {BASELINE_STEP_LIST}")
    print(f"Time Budget: Anchor={MAX_STEP_TIME_ANCHOR}s (Focus on 0.0V)")
    
    for grid_cfg in GRID_LIST:
        for base_step in BASELINE_STEP_LIST:
            print(f"\n>>> Stress Group: {grid_cfg['Tag']} | Step={base_step}V")
            
            for params in SCAN_PARAMS:
                case_id = params['CaseID']
                print(f"  > Case: {case_id} (Width={params['Q_tanh_width']:.2f})", end="")
                
                relay_target = params['RelayBias']
                relay_k = int(np.floor(relay_target / base_step + 1e-9))
                relay_bias_baseline_snapped = relay_k * base_step
                snap_delta = relay_bias_baseline_snapped - relay_target
                relay_meta = {
                    'relay_target': relay_target,
                    'relay_bias_baseline_snapped': relay_bias_baseline_snapped,
                    'relay_bias_a1_start': -1.0, 
                    'relay_bias_a1_delta': 0.0,
                    'relay_delta': snap_delta,
                    'relay_type': 'TARGET' 
                }
                
                # Unconditional Autopsy First 
                print(f"\n    [Strategy] Baseline Diag Check (0.0V, 200 iters)...")
                df_diag, last_phi_base, _, diag_reason, _, _ = run_sweep_stress(
                    params, grid_cfg, base_step, 'Baseline_Diag',
                    start_bias=0.0, stop_bias=0.0,
                    capture_k=None, relay_meta=relay_meta,
                    solver_params=BASELINE_DIAG_PARAMS 
                )
                
                base_status = "FAIL"
                base_fail_class = "NUMERIC_FAIL"
                base_fail_reason = diag_reason
                
                if not df_diag.empty:
                    if df_diag.iloc[-1]['is_converged']:
                         base_status = "CONVERGED"
                         base_fail_class = "CONV"
                         base_fail_reason = "NONE"
                         
                print(f"    [Strategy] Baseline Diag Result: {base_status} ({base_fail_reason})")
                full_logs.append(df_diag)
                
                summary_logs.append({
                    'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                    'solver': 'Baseline_Diag', 'max_bias': 0.0, 'fail_reason': base_fail_reason,
                    'fail_class': base_fail_class, 
                    'total_time': df_diag['time'].sum(),
                    'relay_target': relay_target, 
                    'relay_bias_baseline_snapped': relay_bias_baseline_snapped,
                    'relay_delta': snap_delta
                })

                # A1 Challenge 
                print(f"    [Strategy] A1 Bootstrap Check (0.0V)...")
                
                relay_meta_boot = relay_meta.copy()
                relay_meta_boot['relay_type'] = "BOOTSTRAP_ANCHOR"
                relay_meta_boot['relay_bias_a1_start'] = 0.0
                relay_meta_boot['baseline_fail_class'] = base_fail_class
                relay_meta_boot['baseline_fail_reason'] = base_fail_reason
                
                df_a1_anchor, _, _, fail_r_anchor, _, _ = run_sweep_stress(
                    params, grid_cfg, base_step, 'A1',
                    start_bias=0.0, stop_bias=None,
                    init_phi=None, # Force Ramp Start
                    relay_meta=relay_meta_boot,
                    a1_span=0.0, 
                    solver_params=A1_BOOT_PARAMS
                )
                full_logs.append(df_a1_anchor)
                
                a1_status = "FAIL"
                a1_fail_class = "BOOT_FAIL"
                if not df_a1_anchor.empty:
                    if df_a1_anchor.iloc[-1]['is_converged']:
                        a1_status = "CONVERGED"
                        a1_fail_class = "CONV"
                
                print(f"    [Strategy] A1 Result: {a1_status} ({fail_r_anchor})")

                summary_logs.append({
                    'case_id': case_id, 'grid': grid_cfg['Tag'], 'base_step': base_step,
                    'solver': 'A1', 'max_bias': 0.0, 
                    'fail_reason': 'NONE' if a1_status=="CONVERGED" else fail_r_anchor,
                    'fail_class': a1_fail_class,
                    'total_time': df_a1_anchor['time'].sum(),
                    'relay_type': 'BOOTSTRAP_ANCHOR'
                })

                gc.collect()

    pd.concat(full_logs).to_csv("Stress_v1.4.6-035_FullLog.csv", index=False)
    pd.DataFrame(summary_logs).to_csv("Stress_v1.4.6-035_Summary.csv", index=False)
    print("\n=== STRESS TEST COMPLETE ===")
    print("Saved: Stress_v1.4.6-035_FullLog.csv, Stress_v1.4.6-035_Summary.csv")

if __name__ == "__main__":
    main()