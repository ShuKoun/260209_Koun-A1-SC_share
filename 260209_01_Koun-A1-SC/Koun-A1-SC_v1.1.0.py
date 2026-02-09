import jax
import jax.numpy as jnp
from jax import grad, jit, jvp
from jax.scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import numpy as np
import time

# 強制 64 位精度
jax.config.update("jax_enable_x64", True)

# ============================================================================
# 1. 2D 網格設置 (v1.3.6 Scale Search)
# ============================================================================
print("[Setup] Initializing 2D Grid (v1.3.6 Scale Search)...")

q = 1.602e-19
kb = 1.38e-23
T = 300.0
eps_0 = 8.85e-14
eps_r = 11.7
eps = eps_0 * eps_r
Vt = (kb * T) / q
ni = 1.0e10

Lx = 2.0e-4
Ly = 1.0e-4

# High Res Grid
Nx = 120 
Ny = 60  
# Internal Grid Dimensions
Ny_in = Ny - 2
Nx_in = Nx - 2
N_in = Ny_in * Nx_in

x = jnp.linspace(-Lx/2, Lx/2, Nx)
y = jnp.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = jnp.meshgrid(x, y)

# Doping Profile
Doping_Level = 1.0e16
N_dop = jnp.where(X < 0, Doping_Level, -Doping_Level)
mask_island = (jnp.abs(X) < 0.2e-4) & (jnp.abs(Y - Ly/2) < 0.2e-4)
N_dop = jnp.where(mask_island, Doping_Level * 5.0, N_dop)

# Fixed Boundary Values
phi_bc_L = Vt * jnp.log(Doping_Level/ni)
phi_bc_R = -Vt * jnp.log(Doping_Level/ni)

# Boundary Template
phi_fixed_template = jnp.zeros((Ny, Nx))
phi_fixed_template = phi_fixed_template.at[:, 0].set(phi_bc_L)
phi_fixed_template = phi_fixed_template.at[:, -1].set(phi_bc_R)
for i in range(Nx):
    ratio = i / (Nx - 1)
    val = phi_bc_L * (1 - ratio) + phi_bc_R * ratio
    phi_fixed_template = phi_fixed_template.at[0, i].set(val)
    phi_fixed_template = phi_fixed_template.at[-1, i].set(val)

print(f"[Setup] Active Variables (Internal): {Ny_in}x{Nx_in} ({N_in})")

# ============================================================================
# 2. 物理內核
# ============================================================================

@jit
def reconstruct_full_phi(phi_in):
    phi_full = phi_fixed_template
    phi_in_2d = phi_in.reshape((Ny_in, Nx_in))
    phi_full = phi_full.at[1:-1, 1:-1].set(phi_in_2d)
    return phi_full

@jit
def internal_residual(phi_in):
    phi = reconstruct_full_phi(phi_in)
    n = ni * jnp.exp(phi / Vt)
    p = ni * jnp.exp(-phi / Vt)
    rho = q * (p - n + N_dop)
    
    phi_c = phi[1:-1, 1:-1]
    phi_l = phi[1:-1, :-2]; phi_r = phi[1:-1, 2:]
    phi_u = phi[:-2, 1:-1]; phi_d = phi[2:, 1:-1]
    
    d2x = (phi_r - 2*phi_c + phi_l) / (dx**2)
    d2y = (phi_d - 2*phi_c + phi_u) / (dy**2)
    
    res_in = (d2x + d2y) + rho[1:-1, 1:-1] / eps
    return res_in.flatten()

# ============================================================================
# 3. Koun-A1 治理框架 (v1.3.6)
# ============================================================================

def merit_loss(phi_in):
    res = internal_residual(phi_in)
    return 0.5 * jnp.sum(res**2)

compute_grad_merit = jit(grad(merit_loss))

@jit
def get_diagonal_preconditioner(phi_in):
    phi = reconstruct_full_phi(phi_in)
    phi_c = phi[1:-1, 1:-1]
    n = ni * jnp.exp(phi_c / Vt)
    p = ni * jnp.exp(-phi_c / Vt)
    diag_J = -2.0/(dx**2) - 2.0/(dy**2) - (q / (eps * Vt)) * (n + p)
    return diag_J.flatten()

@jit
def matvec_shifted_J_precond(phi_in, v, dt_inv, M_inv):
    _, Jv = jvp(internal_residual, (phi_in,), (v,))
    Av = v * dt_inv - Jv
    return Av * M_inv

class KounGovernance:
    def __init__(self, max_iter=50, tol=1e-5): 
        self.max_iter = max_iter
        self.tol = tol
        self.MAX_STEP_VOLTAGE = 0.5 
        # PTC Params
        self.dt = 1e-4
        self.dt_max = 1e5
        self.dt_growth = 1.2 
        self.dt_shrink = 0.5
        
    def line_search(self, phi, d, current_merit, atol=1e-9):
        """
        Coherent Dual Tolerance Line Search.
        Accepts if merit_new <= merit_old + atol (Noise/Flat acceptance)
        OR merit_new <= merit_old * (1 - rtol) (Strict descent)
        """
        alpha = 1.0
        rtol = 1e-12 # Minimal strict descent requirement
        
        best_phi = phi; best_merit = current_merit; best_alpha = 0.0
        found_better = False
        
        for i in range(12): 
            step = alpha * d
            # Physical Clamp
            max_v_change = jnp.max(jnp.abs(step))
            physical_scale = 1.0
            if max_v_change > self.MAX_STEP_VOLTAGE:
                physical_scale = self.MAX_STEP_VOLTAGE / max_v_change
                step = step * physical_scale
            
            true_alpha = alpha * physical_scale
            phi_new = phi + step
            merit_new = merit_loss(phi_new)
            
            # Track Best
            if merit_new < best_merit:
                best_merit = merit_new; best_phi = phi_new; best_alpha = true_alpha; found_better = True

            # Acceptance Logic (Dual Tolerance)
            # 1. Strict Descent (Micro)
            if merit_new <= current_merit * (1.0 - rtol):
                 return phi_new, merit_new, true_alpha, True
            
            # 2. Noise Acceptance (Consistent with Governance)
            if merit_new <= current_merit + atol:
                 return phi_new, merit_new, true_alpha, True

            alpha *= 0.5
            
        if found_better:
            return best_phi, best_merit, best_alpha, True 
            
        return phi, current_merit, 0.0, False

    def solve(self, phi_init_in):
        phi_in = phi_init_in
        print(f"\n{'Iter':<5} | {'Int. Res':<10} | {'Lin. Res':<10} | {'dt':<9} | {'Strategy':<15} | {'Alpha'}")
        print("-" * 80)
        
        start_time = time.time()
        
        # Consistent Tolerance for both LS and Governance
        GOV_ATOL = 1e-9
        
        for k in range(self.max_iter):
            res_val = internal_residual(phi_in)
            norm_res = jnp.linalg.norm(res_val)
            merit = 0.5 * norm_res**2
            
            if norm_res < self.tol:
                print(f"{k:<5} | {norm_res:<10.4e} | {'---':<10} | {self.dt:<9.2e} | [CONVERGED]     | ---")
                return reconstruct_full_phi(phi_in)
            
            # 1. PTC Linear Solve (GMRES)
            dt_inv = 1.0 / self.dt
            diag_J = get_diagonal_preconditioner(phi_in)
            M_inv = 1.0 / (dt_inv - diag_J)
            A_op = lambda v: matvec_shifted_J_precond(phi_in, v, dt_inv, M_inv)
            F_precond = res_val * M_inv
            
            d_Newton, info = gmres(A_op, F_precond, tol=1e-2, maxiter=100)
            
            # GMRES Quality Check
            lin_check = A_op(d_Newton) - F_precond
            norm_lin_res = jnp.linalg.norm(lin_check) / (jnp.linalg.norm(F_precond) + 1e-12)
            
            # 2. Try Newton/PTC Step
            # Use same ATOL as Governance check
            phi_new, merit_new, alpha, success = self.line_search(phi_in, d_Newton, merit, atol=GOV_ATOL)
            
            # Classification (Absolute Difference)
            diff = merit_new - merit
            
            # Logic: If LS accepted it, we trust it, UNLESS it's a huge uphill jump that LS missed (unlikely with logic above)
            # We classify based on diff relative to GOV_ATOL
            
            is_good = success and (diff < 0) # Strict descent
            is_noise = success and (diff >= 0 and diff <= GOV_ATOL) # Acceptable noise
            is_uphill = (not success) or (diff > GOV_ATOL) # Bad
            
            if is_good or is_noise:
                status = "[PTC STEP]" if is_good else f"[NOISE {diff:.1e}]"
                phi_in = phi_new
                print(f"{k:<5} | {norm_res:<10.4e} | {norm_lin_res:<10.2e} | {self.dt:<9.2e} | {status:<15} | {alpha:.1e}")
                self.dt = min(self.dt * self.dt_growth, self.dt_max)
            
            else:
                # 3. Rescue Family (Scale Search Sniper)
                if not success: reason = "[REJECT]"
                else: reason = f"[UPHILL {diff:.1e}]"
                
                print(f"{k:<5} | {norm_res:<10.4e} | {norm_lin_res:<10.2e} | {self.dt:<9.2e} | {reason} -> SNIPER| ---")
                
                # Sniper Step: Multi-Scale Search
                g = compute_grad_merit(phi_in)
                norm_g = jnp.linalg.norm(g)
                safe_norm_g = jnp.where(norm_g > 1e-12, norm_g, 1.0)
                
                # Base direction
                d_Base = -g * (norm_res / safe_norm_g)
                
                sniper_scales = [0.1, 0.05, 0.01, 0.005, 0.001]
                sniper_success = False
                
                for scale in sniper_scales:
                    d_Rescue = d_Base * scale
                    # Loose LS for Sniper
                    phi_res, merit_res, alpha_r, success_r = self.line_search(phi_in, d_Rescue, merit, atol=1e-5)
                    
                    if success_r:
                        phi_in = phi_res
                        print(f"{' ':5} | {'(Rescued)':<10} | {'...':<10} | {'...':<9} | [HIT Scale={scale}]| {alpha_r:.1e}")
                        sniper_success = True
                        # If Sniper hits, hold dt (don't grow, don't shrink too much)
                        self.dt *= 0.9 
                        break
                
                if not sniper_success:
                    print(f"{' ':5} | {'(Failed)':<10} | {'...':<10} | {'...':<9} | [SNIPER MISS]   | ---")
                    # Hard shrink
                    self.dt *= self.dt_shrink
                    print(f"{' ':5} | {'...':<10} | {'...':<10} | {self.dt:<9.2e} | [DT SHRINK]     | ---")

        elapsed = time.time() - start_time
        print(f"\n[Summary] Solver finished in {elapsed:.2f}s")
        return reconstruct_full_phi(phi_in)

# ============================================================================
# 4. 運行與可視化
# ============================================================================
if __name__ == "__main__":
    print("[Main] Starting v1.3.6 Scale Search Simulation...")
    phi_init_in = jnp.zeros(N_in)
    
    solver = KounGovernance(max_iter=50, tol=1e-4)
    phi_final = solver.solve(phi_init_in)
    
    print("[Main] Plotting results...")
    phi_2d = phi_final
    
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.pcolormesh(X*1e4, Y*1e4, phi_2d, cmap='viridis', shading='auto')
    fig.colorbar(im1, ax=ax1, label='V')
    ax1.set_title(f'v1.3.6 Potential')
    
    res_flat = internal_residual(phi_final[1:-1, 1:-1].flatten())
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(jnp.log10(jnp.abs(res_flat) + 1e-15), bins=50, color='orange')
    ax2.set_title('Log10(|Int. Res|) Distribution')
    
    ax3 = plt.subplot(1, 3, 3)
    res_full = jnp.zeros((Ny, Nx))
    res_full = res_full.at[1:-1, 1:-1].set(res_flat.reshape((Ny_in, Nx_in)))
    im3 = ax3.pcolormesh(X*1e4, Y*1e4, jnp.log10(jnp.abs(res_full) + 1e-15), cmap='inferno', shading='auto')
    fig.colorbar(im3, ax=ax3, label='Log10(|R|)')
    ax3.set_title('Residual Heatmap')

    plt.tight_layout()
    plt.show()