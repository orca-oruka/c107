import numpy as np
import matplotlib.pyplot as plt

# ==== パラメータ設定 ====
#np.random.seed(0)

DOMAIN    = [0.0, 1.0]   # x,y ∈ [0,1]
GRID_N    = 50           # グリッド分割数
N_SRC     = 6            # 能動源数
N_RCV     = 6            # 受信機数
N_BUOYS   = N_SRC + N_RCV
rho       = 0.15          # Cassini 楕円パラメータ

# グリッド上のターゲット候補点生成
xs, ys = np.linspace(*DOMAIN, GRID_N), np.linspace(*DOMAIN, GRID_N)
Xg, Yg = np.meshgrid(xs, ys)
grid_pts = np.vstack([Xg.ravel(), Yg.ravel()]).T  # shape=(M,2)

def compute_coverage(buoys):
    """覆域率 f_cov を計算"""
    srcs = buoys[:N_SRC]
    rcvs = buoys[N_SRC:]
    covered = np.zeros(len(grid_pts), dtype=bool)
    for s in srcs:
        ds = np.linalg.norm(grid_pts - s, axis=1)
        for r in rcvs:
            dr = np.linalg.norm(grid_pts - r, axis=1)
            covered |= (ds * dr <= rho**2)
    return covered.mean()

def simulated_annealing(
    init_buoys,
    T0=1.0, alpha=0.95,
    n_iter=5000, move_scale=0.05
):
    """
    履歴を記録しながら SA を実行
    returns: best positions, best score, coverage_history
    """
    buoys  = init_buoys.copy()
    best   = buoys.copy()
    f_curr = compute_coverage(buoys)
    f_best = f_curr
    T      = T0

    # 履歴リストに初期値を格納
    history = [f_curr]

    for k in range(n_iter):
        i = np.random.randint(N_BUOYS)
        prop = buoys.copy()
        prop[i] += (np.random.rand(2) - 0.5) * move_scale
        prop[i] = np.clip(prop[i], DOMAIN[0], DOMAIN[1])

        f_prop = compute_coverage(prop)
        delta  = f_prop - f_curr

        if delta >= 0 or np.random.rand() < np.exp(delta / T):
            buoys, f_curr = prop, f_prop
            if f_curr > f_best:
                best, f_best = buoys.copy(), f_curr

        T *= alpha
        history.append(f_curr)

    return best, f_best, history

def avg_nearest_neighbor_distance(pts):
    """同一カテゴリ内の平均最近傍距離を計算"""
    n = len(pts)
    if n < 2:
        return 0.0
    dmat = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
    np.fill_diagonal(dmat, np.inf)
    return dmat.min(axis=1).mean()

def avg_cross_nn_distance(srcs, rcvs):
    """Source→Receiver 間の平均最近傍距離を計算"""
    dmat = np.linalg.norm(srcs[:,None,:] - rcvs[None,:,:], axis=2)
    return dmat.min(axis=1).mean()

def multi_start_sa(num_starts=5, **sa_kwargs):
    """
    複数初期配置から SA を実行し、
    各試行の距離指標・履歴を出力、
    最良解と全履歴を返却
    """
    global_best_score = -np.inf
    global_best_buoys = None
    histories = []

    for i in range(1, num_starts+1):
        init_buoys = np.random.rand(N_BUOYS, 2)
        init_cov   = compute_coverage(init_buoys)

        best_buoys, best_score, history = simulated_annealing(
            init_buoys, **sa_kwargs
        )
        histories.append(history)

        # 距離指標計算
        srcs, rcvs = best_buoys[:N_SRC], best_buoys[N_SRC:]
        mnn_src   = avg_nearest_neighbor_distance(srcs)
        mnn_rcv   = avg_nearest_neighbor_distance(rcvs)
        cross_sr  = avg_cross_nn_distance(srcs, rcvs)

        print(f"[start {i}] init_cov={init_cov:.3f} -> sa_cov={best_score:.3f}")
        print(f"  ① SOURCES avg NN dist   : {mnn_src:.4f}")
        print(f"  ② RECEIVERS avg NN dist : {mnn_rcv:.4f}")
        print(f"  ③ SRC→RCV avg NN dist  : {cross_sr:.4f}\n")

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_buoys = best_buoys.copy()

    return global_best_buoys, global_best_score, histories

if __name__ == "__main__":
    best_buoys, best_score, histories = multi_start_sa(
        num_starts=5,
        T0=0.5, alpha=0.99,
        n_iter=10000, move_scale=0.02
    )

    print(f"=== 全 5 回中の最良覆域: {best_score:.3f} ===")

    # 全 run の履歴をプロット
    plt.figure(figsize=(6,4))
    for idx, hist in enumerate(histories, 1):
        plt.plot(hist, label=f"Run {idx}")
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 最良配置可視化
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(grid_pts[:,0], grid_pts[:,1], s=5, c='lightgray')
    ax.scatter(best_buoys[:N_SRC,0], best_buoys[:N_SRC,1],
               c='C0', label='Source')
    ax.scatter(best_buoys[N_SRC:,0], best_buoys[N_SRC:,1],
               c='C1', label='Receiver')
    ax.set_title(f"Best Coverage = {best_score:.3f}")
    ax.set_xlim(*DOMAIN); ax.set_ylim(*DOMAIN)
    ax.legend()
    plt.show()