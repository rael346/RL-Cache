import pathlib

import numpy as np
import tqdm


def gen_trace_US1_web(
    path: pathlib.Path,
    seed: int = 1234,
    n_requests: int = 10_000_000,
    n_objects: int = 20_000,
    avg_obj_size: int = 20 * 1024,
):
    rng = np.random.default_rng(seed)

    alpha = 1.3
    id_ranges = np.arange(0, n_objects)
    p = 1 / np.power(id_ranges + 1, alpha)
    p /= p.sum()
    obj_ids = rng.choice(id_ranges, size=n_requests, replace=True, p=p)

    mu, sigma = np.log(avg_obj_size), 2.5
    obj_sizes = rng.lognormal(mu, sigma, n_objects)
    while True:
        out_of_range_indices = np.where((obj_sizes < 500) | (obj_sizes > 10**7))[0]
        if len(out_of_range_indices) == 0:
            break
        obj_sizes[out_of_range_indices] = np.random.lognormal(
            mu, sigma, len(out_of_range_indices)
        )
    obj_sizes = obj_sizes.astype(np.int64)

    timestamps = np.cumsum(rng.exponential(0.5, size=n_requests), dtype=np.int64)

    with open(path, "w") as f:
        for i in tqdm.tqdm(range(n_requests), "Generating Trace"):
            f.write(f"{timestamps[i]} {obj_ids[i]} {obj_sizes[obj_ids[i]]}\n")


if __name__ == "__main__":
    trace_path = pathlib.Path("./data/us_web/0.csv")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    gen_trace_US1_web(
        trace_path,
        seed=1234,
        n_requests=14_487_000,
        n_objects=1_111_000,
        avg_obj_size=25 * 1024,
    )
