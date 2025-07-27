import numpy as np
from numpy.linalg import norm, lstsq


# ---------- geometric helpers ----------
def _closest_midpoint(o1, d1, o2, d2):
    w0 = o1 - o2
    a = d1 @ d1
    b = d1 @ d2
    c = d2 @ d2
    d = d1 @ w0
    e = d2 @ w0
    denom = a * c - b * b
    if denom < 1e-12:
        s = 0.0
        t = e / c if c > 1e-12 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return (p1 + p2) * 0.5, norm(p1 - p2)


# ---------- single‑cluster RANSAC ----------
def _ransac_one(origins, dirs, lengths, tol, min_inliers, max_iter, rng,
                min_t, max_t):
    n = origins.shape[0]
    best_pt = None
    best_idx = []

    for _ in range(max_iter):
        i, j = rng.choice(n, 2, replace=False)
        p, sep = _closest_midpoint(origins[i], dirs[i], origins[j], dirs[j])
        if sep > tol:
            continue

        v = p - origins                             # (n,3)
        t = (v * dirs).sum(axis=1)                  # param along each ray
        seg_ok   = (t >= 0) & (t <= lengths)
        range_ok = (t >= min_t) & (t <= max_t)
        dists = norm(v - t[:, None] * dirs, axis=1)  # point–line distance
        inliers = (dists < tol) & seg_ok & range_ok
        idx = np.where(inliers)[0]

        if idx.size > len(best_idx):
            best_pt = p
            best_idx = idx
            if len(best_idx) == n:
                break

    if best_pt is None or len(best_idx) < min_inliers:
        return None, np.empty(0, int)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for k in best_idx:
        d = dirs[k]
        o = origins[k]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    refined, *_ = lstsq(A, b, rcond=None)
    return refined, best_idx


# ---------- multiple clusters ----------
def _ransac_all(rays, tol, min_inliers, max_iter, rng,
                min_t, max_t):
    origins = np.array([r[0] for r in rays], dtype=float)
    dirs = np.array([r[1] - r[0] for r in rays], dtype=float)
    lengths = norm(dirs, axis=1)
    dirs /= lengths[:, None]

    active = np.ones(len(rays), dtype=bool)
    clusters = []

    while active.sum() >= min_inliers:
        idx = np.where(active)[0]
        pt, inliers_local = _ransac_one(
            origins[idx], dirs[idx], lengths[idx],
            tol, min_inliers, max_iter, rng,
            min_t, max_t
        )
        if pt is None:
            break
        global_inliers = idx[inliers_local]
        clusters.append((pt, global_inliers))
        active[global_inliers] = False

    return clusters  # list[(point, ray_indices)]


# ---------- public API ----------
def get_ray_intersections_three_d(
    rays: list,
    *,
    max_ray_distance: float = 100.0,
    min_ray_count: int = 3,
    min_ray_length: float = 100.0,
    max_ray_length: float = 20_000.0,
    method: str = "ransac",
    max_iter: int = 2000,
    rng: np.random.Generator | None = None,
) -> list:
    if rng is None:
        rng = np.random.default_rng()

    if method != "ransac":
        raise ValueError(f"unknown method '{method}'")

    clusters = _ransac_all(
        rays,
        tol=max_ray_distance,
        min_inliers=min_ray_count,
        max_iter=max_iter,
        rng=rng,
        min_t=min_ray_length,
        max_t=max_ray_length,
    )

    return [
        {"point": pt, "rays": np.array(inl, dtype=int)}
        for pt, inl in clusters
    ]
