import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ==== パラメータ設定 ====
np.random.seed(0)

DOMAIN    = [0.0, 1.0]
GRID_N    = 50
N_SRC     = 6
N_RCV     = 6
N_BUOYS   = N_SRC + N_RCV
rho       = 0.2

# グリッド上のターゲット候補点生成
xs, ys   = np.linspace(*DOMAIN, GRID_N), np.linspace(*DOMAIN, GRID_N)
Xg, Yg   = np.meshgrid(xs, ys)
grid_pts = np.vstack([Xg.ravel(), Yg.ravel()]).T  # shape=(M,2)

def compute_coverage(buoys):
    """Cassini 楕円モデルによる覆域率を計算"""
    srcs, rcvs = buoys[:N_SRC], buoys[N_SRC:]
    covered    = np.zeros(len(grid_pts), dtype=bool)
    for s in srcs:
        ds = np.linalg.norm(grid_pts - s, axis=1)
        for r in rcvs:
            dr = np.linalg.norm(grid_pts - r, axis=1)
            covered |= (ds * dr <= rho**2)
    return covered.mean()

def simulated_annealing(init_buoys,
                        T0=0.5, alpha=0.99,
                        n_iter=10000, move_scale=0.02):
    """
    SAを実行し、各ステップ後の配置を履歴に残す
    returns: pos_history (list of (N_BUOYS,2) arrays), cov_history
    """
    buoys     = init_buoys.copy()
    f_curr    = compute_coverage(buoys)
    T         = T0

    pos_history = [buoys.copy()]
    cov_history = [f_curr]

    for k in range(1, n_iter+1):
        # ランダムに1つのブイを微小移動
        i = np.random.randint(N_BUOYS)
        prop = buoys.copy()
        prop[i] += (np.random.rand(2) - 0.5) * move_scale
        prop[i] = np.clip(prop[i], DOMAIN[0], DOMAIN[1])

        f_prop = compute_coverage(prop)
        delta  = f_prop - f_curr

        if delta >= 0 or np.random.rand() < np.exp(delta / T):
            buoys, f_curr = prop, f_prop

        T *= alpha
        pos_history.append(buoys.copy())
        cov_history.append(f_curr)

    return pos_history, cov_history

def multi_start_sa(num_starts=5, **sa_kwargs):
    """
    複数ランダム初期配置で SA を実行し、
    最良スコアの run の履歴を返却
    """
    best_score = -np.inf
    best_pos_hist = None

    for i in range(1, num_starts+1):
        init_b = np.random.rand(N_BUOYS, 2)
        pos_hist, cov_hist = simulated_annealing(init_b, **sa_kwargs)
        score = max(cov_hist)
        print(f"[start {i}] init_cov={cov_hist[0]:.3f} -> best_cov={score:.3f}")
        if score > best_score:
            best_score = score
            best_pos_hist = pos_hist

    return best_pos_hist

if __name__ == "__main__":
    # SA マルチスタート実行
    pos_hist = multi_start_sa(
        num_starts=5,
        T0=0.5, alpha=0.99,
        n_iter=10000, move_scale=0.02
    )

    # 500ステップ毎にフレームを作成
    step_indices = list(range(0, len(pos_hist), 500))
    if step_indices[-1] != len(pos_hist)-1:
        step_indices.append(len(pos_hist)-1)

    # GIF アニメーション作成
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(*DOMAIN)
    ax.set_ylim(*DOMAIN)
    scatter_s = ax.scatter([], [], c='C0', label='Source')
    scatter_r = ax.scatter([], [], c='C1', label='Receiver')
    step_text = ax.text(0.95, 0.05, '', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=12)
    ax.legend(loc='upper right')

    def update(frame_idx):
        step = step_indices[frame_idx]
        buoys = pos_hist[step]
        scatter_s.set_offsets(buoys[:N_SRC])
        scatter_r.set_offsets(buoys[N_SRC:])
        step_text.set_text(f"Step: {step}")
        return scatter_s, scatter_r, step_text

    anim = FuncAnimation(
        fig, update,
        frames=len(step_indices),
        interval=500,
        blit=True
    )

    # PillowWriter で GIF 保存
    writer = PillowWriter(fps=2)
    anim.save("buoy_evolution.gif", writer=writer)
    print("GIF saved as buoy_evolution.gif")
