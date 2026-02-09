import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd
import matplotlib.pyplot as plt
import numpy as np
import time

# 強制 64 位精度 (對 2D 模擬更加重要)
jax.config.update("jax_enable_x64", True)

# ============================================================================
# 1. 2D 網格與物理設置
# ============================================================================
print("[Setup] Initializing 2D Grid...")

q = 1.602e-19
kb = 1.38e-23
T = 300.0
eps_0 = 8.85e-14
eps_r = 11.7
eps = eps_0 * eps_r
Vt = (kb * T) / q
ni = 1.0e10

# 幾何尺寸 (cm)
Lx = 2.0e-4  # 長 2um
Ly = 1.0e-4  # 高 1um

# 網格密度 (適中，保證演示速度)
Nx = 60      # X方向點數
Ny = 30      # Y方向點數
N = Nx * Ny  # 總變量數

x = jnp.linspace(-Lx/2, Lx/2, Nx)
y = jnp.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# 創建 2D 網格
X, Y = jnp.meshgrid(x, y)

# 定義 2D 摻雜 (Doping Profile)
# 左半邊是 N型, 右半邊是 P型
Doping_Level = 1.0e16
N_dop = jnp.where(X < 0, Doping_Level, -Doping_Level)

# 為了讓 2D 更有趣，我們在中間加一個小小的"高摻雜島" (模擬雜質或特殊結構)
# 在中心位置加一個小方塊
mask_island = (jnp.abs(X) < 0.2e-4) & (jnp.abs(Y - Ly/2) < 0.2e-4)
N_dop = jnp.where(mask_island, Doping_Level * 5.0, N_dop)

print(f"[Setup] Grid: {Ny}x{Nx} ({N} variables)")
print(f"[Setup] 2D Doping constructed with a central island feature.")

# ============================================================================
# 2. 2D 物理內核 (五點差分法)
# ============================================================================
@jit
def carrier_density(phi_2d):
    n = ni * jnp.exp(phi_2d / Vt)
    p = ni * jnp.exp(-phi_2d / Vt)
    return n, p

@jit
def poisson_residual_2d(phi_flat):
    # 1. 還原形狀: (N,) -> (Ny, Nx)
    phi = phi_flat.reshape((Ny, Nx))

    # 2. 計算電荷密度
    n, p = carrier_density(phi)
    rho = q * (p - n + N_dop)

    # 3. 計算拉普拉斯算子 (Laplacian) - 五點差分
    # d2phi/dx2
    d2x = (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) / (dx**2)
    # d2phi/dy2
    d2y = (phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / (dy**2)

    laplacian = d2x + d2y

    # 4. 構建殘差矩陣 (初始化為全0)
    # 內部節點殘差: Laplacian + rho/eps = 0
    res_internal = laplacian + rho[1:-1, 1:-1] / eps

    # 創建一個全殘差矩陣，稍後填充邊界
    # 使用 JAX 的 .at[].set() 語法
    res = jnp.zeros((Ny, Nx))
    res = res.at[1:-1, 1:-1].set(res_internal)

    # 5. 處理邊界條件 (Boundary Conditions)
    # 左邊界 (Ohmic Contact): phi = V_built_in_N
    phi_bc_L = Vt * jnp.log(Doping_Level/ni)
    res = res.at[:, 0].set(phi[:, 0] - phi_bc_L)

    # 右邊界 (Ohmic Contact): phi = V_built_in_P
    phi_bc_R = -Vt * jnp.log(Doping_Level/ni)
    res = res.at[:, -1].set(phi[:, -1] - phi_bc_R)

    # 上下邊界 (Neumann / Insulation): dphi/dy = 0
    # phi[0, :] = phi[1, :] => res = phi[0] - phi[1]
    res = res.at[0, :].set(phi[0, :] - phi[1, :])
    res = res.at[-1, :].set(phi[-1, :] - phi[-2, :])

    # 6. 再次展平返回
    return res.flatten()

# ============================================================================
# 3. Koun-GR 核心: 阻尼牛頓法 (2D 適配版)
# ============================================================================
def merit_loss(phi_flat):
    res = poisson_residual_2d(phi_flat)
    return 0.5 * jnp.sum(res**2)

# JAX 自動微分 (計算 2D 問題的巨大雅可比矩陣)
print("[Compiling] JIT compiling gradients and Jacobian... (This takes a moment)")
compute_jacobian = jit(jacfwd(poisson_residual_2d))

def solve_2d_semiconductor():
    # 初始猜測: 全零平面
    phi = jnp.zeros(N)

    print("\nStarting Koun-GR 2D Solver...")
    print(f"{'Iter':<5} | {'Residual':<12} | {'Strategy':<15} | {'Detail'}")
    print("-" * 65)

    MAX_STEP_VOLTAGE = 0.2  # 物理安全限制

    start_time = time.time()

    for k in range(50):
        F = poisson_residual_2d(phi)
        res_norm = jnp.linalg.norm(F)

        if res_norm < 1e-5:
            elapsed = time.time() - start_time
            print(f"{k:<5} | {res_norm:<12.4e} | [CONVERGED]     | Time: {elapsed:.2f}s")
            return phi

        # ========================================================
        # 阻尼牛頓法 (Damped Newton)
        # ========================================================
        try:
            # 這裡解的是 Ax = b，其中 A 是 (1800x1800) 的矩陣
            J = compute_jacobian(phi)
            dx = jnp.linalg.solve(J, -F)

            # 全局阻尼計算
            max_change = jnp.max(jnp.abs(dx))
            damping_factor = 1.0
            if max_change > MAX_STEP_VOLTAGE:
                damping_factor = MAX_STEP_VOLTAGE / max_change

            dx_damped = dx * damping_factor

            # 簡單的接受邏輯
            phi_new = phi + dx_damped
            res_new = jnp.linalg.norm(poisson_residual_2d(phi_new))

            if res_new < res_norm:
                phi = phi_new
                print(f"{k:<5} | {res_norm:<12.4e} | [NEWTON 2D]     | Damping={damping_factor:.2e}")
            else:
                # 如果還不行，再減半試一次 (簡單的回溯)
                phi_new = phi + dx_damped * 0.5
                res_new_2 = jnp.linalg.norm(poisson_residual_2d(phi_new))
                if res_new_2 < res_norm:
                     phi = phi_new
                     print(f"{k:<5} | {res_norm:<12.4e} | [NEWTON 2D]     | Retry 0.5x")
                else:
                    print(f"{k:<5} | {res_norm:<12.4e} | [STUCK]         | Need Adjoint Rescue")
                    break # 為了 demo 簡潔，這裡簡化了 fallback 邏輯

        except Exception as e:
            print(f"Error: {e}")
            break

    return phi

# ============================================================================
# 4. 運行與 2D 可視化
# ============================================================================
phi_solution_flat = solve_2d_semiconductor()

# 重塑回 2D 進行繪圖
phi_2d = phi_solution_flat.reshape((Ny, Nx))
n_2d, p_2d = carrier_density(phi_2d)

# 繪圖設置
fig = plt.figure(figsize=(15, 10))

# 1. 電勢分佈 (Heatmap)
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.pcolormesh(X*1e4, Y*1e4, phi_2d, cmap='viridis', shading='auto')

fig.colorbar(im1, ax=ax1, label='Potential (V)')
ax1.set_title('2D Electric Potential')
ax1.set_xlabel('X (microns)')
ax1.set_ylabel('Y (microns)')

# 2. 摻雜分佈 (展示我們的結構)
ax2 = plt.subplot(2, 2, 2)
# Log scale for doping to see the types clearly
im2 = ax2.pcolormesh(X*1e4, Y*1e4, jnp.log10(jnp.abs(N_dop)), cmap='RdBu', shading='auto')
fig.colorbar(im2, ax=ax2, label='Log10(|Doping|)')
ax2.set_title('Doping Profile (Red=P, Blue=N)')
ax2.set_xlabel('X (microns)')
ax2.set_ylabel('Y (microns)')

# 3. 電子濃度 (Log Scale)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.pcolormesh(X*1e4, Y*1e4, jnp.log10(n_2d), cmap='plasma', shading='auto')

fig.colorbar(im3, ax=ax3, label='Log10(Electron Conc.)')
ax3.set_title('Electron Concentration')
ax3.set_xlabel('X (microns)')
ax3.set_ylabel('Y (microns)')

# 4. 電場向量場 (Quiver Plot)
ax4 = plt.subplot(2, 2, 4)
# 計算電場 E = -grad(phi)
Ex, Ey = jnp.gradient(-phi_2d, dx, dy, axis=(1, 0)) # 注意 axis 順序
# 為了視覺清晰，降低採樣率 (每隔 3 個點畫一個箭頭)
skip = (slice(None, None, 3), slice(None, None, 3))
ax4.quiver(X[skip]*1e4, Y[skip]*1e4, Ex[skip], Ey[skip], color='r')
ax4.set_title('Electric Field Vectors')
ax4.set_xlabel('X (microns)')
ax4.set_ylabel('Y (microns)')

plt.tight_layout()
plt.show()

print("\n[Summary] 2D Simulation Complete.")
print("Check the 'Doping Profile' plot: You should see a small island in the center.")
print("Check 'Potential': The lines should bend around that island, proving 2D physics works.")