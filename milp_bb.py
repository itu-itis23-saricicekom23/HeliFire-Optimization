import time as tm
import pandas as pd
import geopandas as gpd
from pulp import (
    LpProblem, LpVariable, LpBinary, LpInteger,
    LpMinimize, lpSum, GLPK_CMD, LpStatus, value,
)
from pulp import GLPK_CMD


BUDGET               = 311  
MAX_STATION_CAPACITY = 1      
MAX_ASSIGN_PER_HELI  = 20      
DISTANCE_LIMIT_KM    = 150.0    

HELI_TYPES = {
    "Type_1": {"cost": 30, "water": 700},
    "Type_2": {"cost": 10, "water": 300},
    "Type_3": {"cost":  4, "water": 180},
}

def load_data():
    print("Loading data...")

    df_risk        = pd.read_csv("Rj_Risk_Matrisi.csv")
    risk_by_grid   = df_risk.set_index("Grid_ID")["R_j"].to_dict()
    grid_ids       = sorted(risk_by_grid.keys())

    df_dist        = pd.read_csv("Dij_Mesafe_Matrisi.csv")
    dist_by_pair   = {
        (r.Istasyon_ID, r.Grid_ID): r.D_ij_km
        for r in df_dist.itertuples(index=False)
    }
    station_ids    = sorted(df_dist["Istasyon_ID"].unique())

    print(f"  Stations : {len(station_ids)}")
    print(f"  Grids    : {len(grid_ids)}")
    print(f"  Risky grids (R_j > 0): {sum(1 for v in risk_by_grid.values() if v > 0)}")

    return station_ids, grid_ids, risk_by_grid, dist_by_pair


def build_milp(station_ids, grid_ids, risk_by_grid, dist_by_pair,
               model_name="HeliFire_MILP"):

    model = LpProblem(model_name, LpMinimize)

    x = {
        (i, k): LpVariable(f"x_{i}_{k}", lowBound=0, upBound=MAX_STATION_CAPACITY,
                            cat=LpInteger)
        for i in station_ids
        for k in HELI_TYPES
    }

    z = {
        (i, j): LpVariable(f"z_{i}_{j}", cat=LpBinary)
        for i in station_ids
        for j in grid_ids
    }

    model += lpSum(
        risk_by_grid[j] * dist_by_pair[(i, j)] * z[(i, j)]
        for i in station_ids
        for j in grid_ids
        if risk_by_grid.get(j, 0) > 0 and (i, j) in dist_by_pair
    ), "Minimize_RiskWeightedDistance"

    for j in grid_ids:
        if risk_by_grid.get(j, 0) > 0:
            model += (
                lpSum(z[(i, j)] for i in station_ids) == 1,
                f"C1_Coverage_{j}"
            )
        else:
            for i in station_ids:
                model += z[(i, j)] == 0, f"C1_NoRisk_{i}_{j}"

    model += (
        lpSum(HELI_TYPES[k]["cost"] * x[(i, k)]
              for i in station_ids for k in HELI_TYPES) <= BUDGET,
        "C2_Budget"
    )

    for i in station_ids:
        model += (
            lpSum(x[(i, k)] for k in HELI_TYPES) <= MAX_STATION_CAPACITY,
            f"C3_Capacity_{i}"
        )

    for i in station_ids:
        model += (
            lpSum(z[(i, j)] for j in grid_ids)
            <= MAX_ASSIGN_PER_HELI * lpSum(x[(i, k)] for k in HELI_TYPES),
            f"C4_AssignCap_{i}"
        )

    for i in station_ids:
        for j in grid_ids:
            model += (
                z[(i, j)] <= lpSum(x[(i, k)] for k in HELI_TYPES),
                f"C5_Activation_{i}_{j}"
            )
            d = dist_by_pair.get((i, j), float("inf"))
            if d > DISTANCE_LIMIT_KM:
                model += z[(i, j)] == 0, f"C6_Range_{i}_{j}"

    return model, x, z


def print_results(model, x, z, station_ids, grid_ids,
                  risk_by_grid, dist_by_pair, exec_time, label="MILP"):

    status = LpStatus[model.status]
    obj    = value(model.objective) if model.status == 1 else None

    print(f"\n{'='*58}")
    print(f"  {label} RESULT")
    print(f"{'='*58}")
    print(f"  Status          : {status}")
    print(f"  Computing Time  : {exec_time:.3f} seconds")

    if model.status != 1:
        print("  No optimal solution found.")
        return

    print(f"  Objective Value : {obj:.4f}")

    total_spent = 0
    total_water = 0
    active_stations = []

    for i in station_ids:
        for k in HELI_TYPES:
            cnt = int(round(value(x[(i, k)]) or 0))
            if cnt > 0:
                total_spent += cnt * HELI_TYPES[k]["cost"]
                total_water += cnt * HELI_TYPES[k]["water"]
                active_stations.append((i, k, cnt))

    assigned_grids   = []
    unassigned_risky = []
    for j in grid_ids:
        if risk_by_grid.get(j, 0) <= 0:
            continue
        covered = any(
            round(value(z[(i, j)]) or 0) == 1 for i in station_ids
        )
        if covered:
            assigned_grids.append(j)
        else:
            unassigned_risky.append(j)

    print(f"  Budget Spent    : ${total_spent}M / ${BUDGET}M")
    print(f"  Total Water Cap : {total_water} gallons")
    print(f"  Active Stations : {len(active_stations)}")
    print(f"  Assigned Grids  : {len(assigned_grids)}")
    print(f"  Unassigned Risky: {len(unassigned_risky)}")

    print(f"\n{'─'*58}")
    print(f"  HELICOPTER ALLOCATION")
    print(f"{'─'*58}")
    print(f"  {'Station':<10} {'Type':<10} {'Water (gal)':<14} {'Cost ($M)'}")

    type_totals = {k: 0 for k in HELI_TYPES}
    for i, k, cnt in sorted(active_stations, key=lambda r: r[0]):
        w = HELI_TYPES[k]["water"] * cnt
        c = HELI_TYPES[k]["cost"]  * cnt
        print(f"  {i:<10} {k:<10} {w:<14} {c}")
        type_totals[k] += cnt

    print(f"\n  Summary by type:")
    for k, total in type_totals.items():
        print(f"    {k}: {total} unit(s) — "
              f"{total * HELI_TYPES[k]['water']} gal — "
              f"${total * HELI_TYPES[k]['cost']}M")

    print(f"{'='*58}\n")



def run_milp(station_ids, top_grids, risk_by_grid, dist_by_pair,
             time_limit=300):
    print(f"\n{'─'*58}")
    print(f"  PART 1: MILP (GLPK solver, time limit={time_limit}s)")
    print(f"  {len(station_ids)} stations × {len(top_grids)} risky grids")
    print(f"{'─'*58}")

    model, x, z = build_milp(station_ids, top_grids, risk_by_grid, dist_by_pair,
                              model_name="HeliFire_MILP")

    print(f"  Variables  : {len(model.variables())}")
    print(f"  Constraints: {len(model.constraints)}")
    print("  Solving...")

    t0 = tm.perf_counter()
    model.solve(GLPK_CMD(msg=True, timeLimit=time_limit))
    elapsed = tm.perf_counter() - t0

    print_results(model, x, z, station_ids, top_grids,
                  risk_by_grid, dist_by_pair, elapsed, label="MILP")

    return model, x, z



def run_branch_and_bound(station_ids, top_grids, risk_by_grid, dist_by_pair, time_limit=120):

    print(f"\n{'─'*58}")
    print(f"  PART 2: BRANCH & BOUND SCALABILITY BENCHMARK")
    print(f"  (Same MILP model, increasing problem size)")
    print(f"{'─'*58}\n")

    scenarios = [10, 25, 50, 100, len(top_grids)]
    scenarios = sorted(set(s for s in scenarios if s <= len(top_grids)))

    results = []

    for n in scenarios:
        sub_grids = top_grids[:n]
        print(f"  → Solving: {len(station_ids)} stations × {n} grids ...", end=" ", flush=True)

        model, x, z = build_milp(
            station_ids, sub_grids, risk_by_grid, dist_by_pair,
            model_name=f"BB_{n}grids"
        )

        t0     = tm.perf_counter()
        model.solve(GLPK_CMD(msg=True, timeLimit=time_limit))
        elapsed = tm.perf_counter() - t0

        status = LpStatus[model.status]
        obj    = round(value(model.objective), 4) if model.status == 1 else None

        n_heli = sum(
            int(round(value(x[(i, k)]) or 0))
            for i in station_ids for k in HELI_TYPES
        ) if model.status == 1 else None

        results.append({
            "Grids": n,
            "Variables": len(model.variables()),
            "Constraints": len(model.constraints),
            "Status": status,
            "Objective": obj,
            "Helis": n_heli,
            "Time (s)": round(elapsed, 3),
        })

        print(f"done in {elapsed:.3f}s — {status}")

    print(f"\n{'─'*58}")
    print(f"  BRANCH & BOUND SCALABILITY TABLE")
    print(f"{'─'*58}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


    return df


if __name__ == "__main__":
    station_ids, grid_ids, risk_by_grid, dist_by_pair = load_data()

    top_grids = [j for j in grid_ids if risk_by_grid.get(j, 0) > 0]

    print(f"\nProblem scope: {len(station_ids)} stations, {len(top_grids)} risky grids")
    print(f"Budget: ${BUDGET}M | Capacity: {MAX_STATION_CAPACITY} heli/pad | Range: {DISTANCE_LIMIT_KM} km\n")

    milp_model, x, z = run_milp(
        station_ids, top_grids, risk_by_grid, dist_by_pair,
        time_limit=300
    )

    bb_results = run_branch_and_bound(
        station_ids, top_grids, risk_by_grid, dist_by_pair
    )

    bb_results.to_csv("BB_Benchmark_Results.csv", index=False)
    print("BB_Benchmark_Results.csv saved.")