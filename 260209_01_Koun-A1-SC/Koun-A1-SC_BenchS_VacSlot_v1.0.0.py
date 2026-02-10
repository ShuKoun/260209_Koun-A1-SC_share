"""
文件名 (Filename): Koun-A1-SC_BenchS_VacSlot_v1.7.12.py
中文標題 (Chinese Title): [Benchmark S v1.7.12] 黃金數據源定稿
英文標題 (English Title): [Benchmark S v1.7.12] The Definitive Golden Source
版本號 (Version): v1.7.12 (Golden Source)
前置版本 (Prev Version): v1.7.11 (Instrumentation Polish)

變更日誌 (Changelog):
    1. [Data] 字段重命名：is_relay -> is_relay_mode，消除與 gmres_boosted 的語義混淆。
    2. [Data] 成功標記：新增 is_converged (bool) 列，簡化後續繪圖過濾邏輯。
    3. [Meta] 核心鎖定：數值計算邏輯嚴格保持 v1.7.7 狀態，僅優化數據字典。
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
# 0. Configuration
# ============================================================================
q = 1.602e-19; kb = 1.38e-23; T = 300.0; eps_0 = 8.85e-14
Vt = (kb * T) / q
ni = 1.0e10; ni_vac = 1.0e-20

Lx = 1.0e-5; Ly = 0.5e-5
COMMON_GRID = {'Nx': 120, 'Ny': 60}

SCAN_PARAMS = [
    # Case 1: Symmetric Sanity (5nm, 3e20/3e20)
    {'CaseID': 'C1', 'SlotW_nm': 5.0, 'N_high': 3e20, 'N_low': 3e20, 'BiasMax': 2.0, 'Q_trap': 0.0, 'Step': 0.2, 'Alpha': 0.0, 'RelayBias': 0.0, 'A1_Step': 0.2},

    # Case 2: Asymmetric (2nm, 3e20/1e17)
    {'CaseID': 'C2', 'SlotW_nm': 2.0, 'N_high': 3e20, 'N_low': 1e17, 'BiasMax': 8.0, 'Q_trap': 0.0, 'Step': 0.2, 'Alpha': 0.4, 'RelayBias': 7.0, 'A1_Step': 0.1},

    # Case 3: The Killer (2nm, 1e21/1e17 + Trap 1e18) - 物理參數凍結
    {'CaseID': 'C3', 'SlotW_nm': 2.0, 'N_high': 1e21, 'N_low': 1e17, 'BiasMax': 8.0, 'Q_trap': 1.0e18, 'Step': 0.2, 'Alpha': 0.2, 'RelayBias': 4.0, 'A1_Step': 0.05},
]

MAX_STEP_TIME = 60.0

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
# 1. Kernels (JIT) - Identical to v1.7.7
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
# 2. Solvers (Instrumented v1.7.12)
# ============================================================================
class BaselineNewton:
    def __init__(self, params): self.params = params

    def solve_step(self, phi_init, bias_L, bias_R, physics_args):
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

        # Baseline uses fixed GMRES params
        g_tol, g_max, g_rst = self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart']

        for k in range(self.params['max_iter']):
            if time.time() - start_time > MAX_STEP_TIME:
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

    def solve_step(self, phi_init, bias_L, bias_R, prev_dt, physics_args, is_relay_hard=False):
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

        # [v1.7.12] Initialize capture variable
        captured_g_params = (self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart'])

        for k in range(self.params['max_outer_iter']):
            if time.time() - start_time > MAX_STEP_TIME:
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

            # [v1.7.12] Logic Identical to v1.7.7 (Numerically Identical Core)
            if is_relay_hard and norm_init > 1e1:
                tol_g, max_g, rst_g = 1e-2, 80, 20
            else:
                tol_g, max_g, rst_g = self.params['gmres_tol'], self.params['gmres_maxiter'], self.params['gmres_restart']

            # Capture actual params used
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
# 3. Execution (Deep Instrumentation v1.7.12)
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

def run_sweep(params, solver_type='A1', bias_limit=None, start_bias=0.0, init_phi=None, custom_step=None):
    print(f"\n[{solver_type}] Start: Case={params.get('CaseID','?')}, Slot={params['SlotW_nm']}nm, Nh={params['N_high']:.0e}, Trap={params.get('Q_trap', 0):.0e}")
    X, Y, dx, dy = setup_grid(COMMON_GRID['Nx'], COMMON_GRID['Ny'])
    eps_map, N_dop, ni_map, Q_trap_map = setup_materials(X, Y, params)
    physics_args = (eps_map, N_dop, ni_map, Q_trap_map, dx, dy, int(COMMON_GRID['Nx']), int(COMMON_GRID['Ny']))

    phi_bc_L = Vt * jnp.log(params['N_high']/ni)
    phi_bc_R_phys = Vt * jnp.log(params['N_low']/ni)
    alpha = params.get('Alpha', 0.0)
    phi_bc_R_base = phi_bc_L + alpha * (phi_bc_R_phys - phi_bc_L)

    if init_phi is not None:
        phi_init = jnp.array(init_phi)
    else:
        phi_seed_L = phi_bc_L
        phi_seed_R = phi_bc_R_phys
        phi_full = jnp.zeros((COMMON_GRID['Ny'], COMMON_GRID['Nx']))
        for i in range(COMMON_GRID['Nx']):
            r = i/(COMMON_GRID['Nx']-1)
            phi_full = phi_full.at[:, i].set(phi_seed_L*(1-r) + phi_seed_R*r)
        phi_init = phi_full[1:-1, 1:-1].flatten()

    solver = KounA1Solver(A1_PARAMS) if solver_type == 'A1' else BaselineNewton(BASELINE_PARAMS)
    current_dt = 1e-4

    actual_max = bias_limit if bias_limit is not None else params['BiasMax']
    step_size = custom_step if custom_step else params.get('Step', 0.5)

    start_bias_raw = start_bias
    start_k = int(np.floor(start_bias_raw / step_size + 1e-9))
    start_bias = start_k * step_size

    if solver_type == 'A1' and init_phi is not None:
        delta = start_bias - start_bias_raw
        print(f"# RelaySnap: raw={start_bias_raw:.4f}, snapped={start_bias:.4f}, step={step_size:.4f}, delta={delta:.1e}")
        print(f"# A1_dt: max={A1_PARAMS['dt_max']}, growth_cap={A1_PARAMS['dt_growth_cap']}, shrink_noise={A1_PARAMS['dt_shrink_noise']}")

    end_k = int(np.floor(actual_max / step_size + 1e-9))
    k_steps = np.arange(start_k, end_k + 1)

    print("bias,success,res0,res1,rel,iters,dt,status,gmres,time,t_lin,t_res,t_ls,d_phi,g_tol,g_max,g_rst")
    results = []

    history_phi = {}
    is_relay_run = (solver_type == 'A1' and init_phi is not None)
    relay_dt_boost_done = False

    step_cnt = 0
    for k in k_steps:
        bias = k * step_size
        bc_R = phi_bc_R_base + bias
        start = time.time()

        res0_vec = internal_residual(phi_init, phi_bc_L, bc_R, *physics_args)
        res0 = float(jnp.linalg.norm(res0_vec))

        a1_counts = {'step':0, 'noise':0, 'sniper':0}

        if solver_type == 'A1':
            phi_next, succ, _, res1, rel, iters, extra, last_gmres, next_dt, t_lin, t_res, t_ls, a1_counts, g_params = solver.solve_step(phi_init, phi_bc_L, bc_R, current_dt, physics_args, is_relay_hard=is_relay_run)
            current_dt = next_dt

            if is_relay_run and (not relay_dt_boost_done) and iters == 0:
                current_dt = 1e-3
                relay_dt_boost_done = True
        else:
            phi_next, succ, _, res1, rel, iters, extra, last_gmres, t_lin, t_res, t_ls, g_params = solver.solve_step(phi_init, phi_bc_L, bc_R, physics_args)

        dur = time.time() - start
        d_phi = float(jnp.max(jnp.abs(phi_next - phi_init))) if succ else 0.0

        row_str = f"{bias:.2f},{int(succ)},{res0:.2e},{res1:.2e},{rel:.2e},{iters},{float(current_dt) if solver_type=='A1' else 0.0:.1e},{extra},{last_gmres},{dur:.2f},{t_lin:.2f},{t_res:.2f},{t_ls:.2f},{d_phi:.2e},{g_params[0]},{g_params[1]},{g_params[2]}"
        print(row_str)
        sys.stdout.flush()

        # [Instr v1.7.12] Status Normalization & Strict Boost
        raw_status = extra.split('(')[0] if '(' in extra else extra
        status_kind = 'CONV' if raw_status == 'CONVERGED' else raw_status
        is_converged = (status_kind == 'CONV')

        # Strict Tuple Comparison
        gmres_boosted = (g_params == (1e-2, 80, 20))

        results.append({
            'solver': solver_type,
            'case_id': params.get('CaseID', 'CX'),
            'slot_nm': params['SlotW_nm'],
            'trap': params.get('Q_trap', 0.0),
            'N_high': params['N_high'],
            'N_low': params['N_low'],
            'alpha': params.get('Alpha', 0.0),
            'bias_max': params['BiasMax'],
            'relay_bias': params.get('RelayBias', 0.0),
            'step_size': step_size,
            'bias': bias,
            'status': extra,
            'status_kind': status_kind,
            'is_converged': is_converged, # Added v1.7.12
            'success': succ,
            'iters': iters,
            'dt': float(current_dt) if solver_type=='A1' else 0.0,
            'time': dur,
            't_lin': t_lin, 't_res': t_res, 't_ls': t_ls,
            'res0': res0,
            'res1': res1,
            'rel': rel,
            'd_phi': d_phi,
            'is_relay_mode': is_relay_run, # Renamed v1.7.12
            'gmres_boosted': gmres_boosted,
            'a1_cnt_step': a1_counts['step'],
            'a1_cnt_noise': a1_counts['noise'],
            'a1_cnt_sniper': a1_counts['sniper'],
            'g_tol': g_params[0],
            'g_max': g_params[1],
            'g_rst': g_params[2]
        })

        del res0_vec
        if succ:
            history_phi[int(k)] = np.array(phi_next)
            del phi_init
            phi_init = phi_next
            del phi_next
        else:
            print(f"# FAILED at {bias:.1f}V")
            gc.collect()
            break

        step_cnt += 1
        if step_cnt % 5 == 0: gc.collect()

    return pd.DataFrame(results), history_phi

def main():
    all_res = []

    for case_idx, params in enumerate(SCAN_PARAMS):
        print(f"\n# === PROCESSING {params['CaseID']} ===")

        # 1. Run Baseline
        df_base, hist_base = run_sweep(params, 'Baseline')
        all_res.append(df_base)

        base_max_bias = df_base['bias'].max()
        base_failed = not df_base.iloc[-1]['success']

        # 2. Run A1 (Relay Mode)
        relay_bias_cfg = params.get('RelayBias', 0.0)
        base_step = params.get('Step', 0.5)
        a1_step = params.get('A1_Step', base_step)

        target_k_base = int(np.floor(relay_bias_cfg / base_step + 1e-9))
        available_k_base = sorted(hist_base.keys())
        candidates = [k for k in available_k_base if k <= target_k_base]

        if not candidates:
            print(f"Warning: No valid relay candidates <= {relay_bias_cfg}V. Skipping A1.")
            continue

        start_k_base = max(candidates)
        start_bias = start_k_base * base_step
        init_phi = hist_base[start_k_base]

        if not base_failed:
             a1_limit = start_bias + 1.0
             if a1_limit > params['BiasMax']: a1_limit = params['BiasMax']
             if case_idx > 0:
                 print(f"\n>>> Baseline SUCCEEDED up to {base_max_bias}V. Running A1 Relay from {start_bias}V to {a1_limit}V...")
             else:
                 print(f"\n>>> Baseline SUCCEEDED (Case 1). Running A1 Relay from {start_bias}V to {a1_limit}V...")
        else:
             a1_limit = base_max_bias + 0.4
             print(f"\n>>> Baseline FAILED at {base_max_bias}V. Running A1 Relay from {start_bias}V to {a1_limit}V...")

        df_a1, _ = run_sweep(params, 'A1', bias_limit=a1_limit, start_bias=start_bias, init_phi=init_phi, custom_step=a1_step)
        all_res.append(df_a1)

        gc.collect()
        try: jax.clear_caches()
        except: pass

    full = pd.concat(all_res)
    full.to_csv("BenchS_v1712_FullLog.csv", index=False)
    print("\n=== SUMMARY TABLE ===")
    summary = full.groupby(['solver', 'case_id']).agg(
        max_bias=('bias', 'max'),
        fail_reason=('status', 'last')
    )
    print(summary)
    print("\n>>> Log saved to BenchS_v1712_FullLog.csv")

if __name__ == "__main__":
    main()