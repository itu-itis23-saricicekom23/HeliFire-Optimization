import pandas as pd
import geopandas as gpd
from pulp import (
    LpProblem,
    LpVariable,
    LpBinary,
    LpInteger,
    LpMinimize,
    lpSum,
    PULP_CBC_CMD,
    value,
)

# --- PARAMETRELER VE BUTCE ---
BUDGET = 311
MAX_ASSIGNMENTS_PER_HELI = 20  # Esnetildi: Daha az istasyon, daha cok butce kalsin diye
MAX_STATION_CAPACITY = 1
DISTANCE_LIMIT_KM = 150.0  # Esnetildi

HELI_TYPES = {
    "Type_1": {"cost": 30, "water": 700},
    "Type_2": {"cost": 10, "water": 300},
    "Type_3": {"cost": 4, "water": 180},
}


def load_data(
    distance_csv="Dij_Mesafe_Matrisi.csv",
    risk_csv="Rj_Risk_Matrisi.csv",
    helipads_geojson="helipads_clean.geojson",
    su_geojson="su.geojson",
    grid_csv="Grid_Coordinates.csv",
):
    df_risk = pd.read_csv(risk_csv)
    grid_ids = sorted(df_risk["Grid_ID"].unique())
    risk_by_grid = df_risk.set_index("Grid_ID")["R_j"].to_dict()

    gdf_stations = gpd.read_file(helipads_geojson, on_invalid="ignore")
    station_ids = sorted(gdf_stations["Istasyon_ID"].unique())

    gdf_su = gpd.read_file(su_geojson, on_invalid="ignore").to_crs(epsg=32635)
    gdf_stations = gdf_stations.to_crs(epsg=32635)

    df_grid = pd.read_csv(grid_csv)
    gdf_grid = gpd.GeoDataFrame(
        df_grid, geometry=gpd.points_from_xy(df_grid.lon, df_grid.lat), crs="EPSG:4326"
    ).to_crs(epsg=32635)

    istasyon_su_join = gpd.sjoin_nearest(
        gdf_stations, gdf_su, how="left", distance_col="s_i"
    )
    s_i_df = istasyon_su_join.groupby("Istasyon_ID")["s_i"].min().reset_index()

    grid_su_join = gpd.sjoin_nearest(gdf_grid, gdf_su, how="left", distance_col="w_j")
    w_j_df = grid_su_join.groupby("Grid_ID")["w_j"].min().reset_index()

    alpha = 5
    dist_by_pair = {}
    w_j_dict = {}
    for i_row in s_i_df.itertuples():
        for j_row in w_j_df.itertuples():
            D_ij_km = (i_row.s_i + (alpha + 1) * j_row.w_j) / 1000.0
            dist_by_pair[(i_row.Istasyon_ID, j_row.Grid_ID)] = D_ij_km
            w_j_dict[j_row.Grid_ID] = j_row.w_j

    return (
        station_ids,
        grid_ids,
        risk_by_grid,
        dist_by_pair,
        w_j_dict,
        s_i_df.set_index("Istasyon_ID")["s_i"].to_dict(),
    )


def build_milp(station_ids, grid_ids, risk_by_grid, dist_by_pair):
    model = LpProblem("Week2_Core_MILP", LpMinimize)

    x = {
        (i, k): LpVariable(f"x_{i}_{k}", lowBound=0, cat=LpInteger)
        for i in station_ids
        for k in HELI_TYPES
    }
    z = {
        (i, j): LpVariable(f"z_{i}_{j}", cat=LpBinary)
        for i in station_ids
        for j in grid_ids
    }

    model += (
        lpSum(
            risk_by_grid[j] * dist_by_pair[(i, j)] * z[(i, j)]
            for i in station_ids
            for j in grid_ids
            if risk_by_grid[j] > 0
        )
        - 0.001
        * lpSum(
            HELI_TYPES[k]["water"] * x[(i, k)] for i in station_ids for k in HELI_TYPES
        ),
        "Minimize_Distance_And_Maximize_Water",
    )

    for j in grid_ids:
        if risk_by_grid[j] > 0:
            model += (
                lpSum(z[(i, j)] for i in station_ids) == 1,
                f"Assign_Grid_{j}_Exactly_Once",
            )
        else:
            model += (
                lpSum(z[(i, j)] for i in station_ids) == 0,
                f"Ignore_Risk_Free_Grid_{j}",
            )

    for i in station_ids:
        model += (
            lpSum(x[(i, k)] for k in HELI_TYPES) <= MAX_STATION_CAPACITY,
            f"StationCap_{i}",
        )

    model += (
        lpSum(
            HELI_TYPES[k]["cost"] * x[(i, k)] for i in station_ids for k in HELI_TYPES
        )
        <= BUDGET,
        "BudgetConstraint",
    )

    for i in station_ids:
        model += (
            lpSum(z[(i, j)] for j in grid_ids)
            <= MAX_ASSIGNMENTS_PER_HELI * lpSum(x[(i, k)] for k in HELI_TYPES),
            f"AssignmentCapacity_{i}",
        )

    for i in station_ids:
        for j in grid_ids:
            model += (
                z[(i, j)] <= lpSum(x[(i, k)] for k in HELI_TYPES),
                f"ActivationLink_{i}_{j}",
            )
            if dist_by_pair[(i, j)] > DISTANCE_LIMIT_KM:
                model += z[(i, j)] == 0, f"DistanceLimit_{i}_{j}"

    return model, x, z


def analyze_station_decisions(
    model, x, z, station_ids, grid_ids, risk_by_grid, dist_by_pair, s_i_dict
):
    print("\n=== ISTASYON KARAR ANALIZI ===")
    print(
        f"{'Istasyon':<10} {'Suya Uzaklik(km)':<18} {'Helikopterler':<30} {'Kapsanan Risk':<15} {'Ort. Mesafe':<15} {'Atanan Grid#':<15}"
    )
    print("=" * 110)

    selected_stations = []
    for i in station_ids:
        heli_count = sum(int(x[(i, k)].value() or 0) for k in HELI_TYPES)
        assigned_grids = [j for j in grid_ids if z[(i, j)].value() == 1]
        total_risk = sum(risk_by_grid[j] for j in assigned_grids)

        if len(assigned_grids) > 0:
            avg_distance = sum(dist_by_pair[(i, j)] for j in assigned_grids) / len(
                assigned_grids
            )
        else:
            avg_distance = 0

        s_i = s_i_dict.get(i, 0) / 1000

        if heli_count > 0:
            heli_details = ", ".join(
                [
                    f"{k}:{int(x[(i, k)].value())}"
                    for k in HELI_TYPES
                    if x[(i, k)].value() > 0
                ]
            )

            selected_stations.append((i, heli_count, total_risk, s_i))
            print(
                f"{i:<10} {s_i:<18.2f} {heli_details:<30} {total_risk:<15.3f} {avg_distance:<15.2f} {len(assigned_grids):<15}"
            )

    print(f"\n>>> TOPLAM SECILEN ISTASYON SAYISI: {len(selected_stations)}")


def summarize_solution(model, x, z, station_ids, grid_ids, risk_by_grid, dist_by_pair):
    if model.status != 1:
        print(
            "\n[HATA] Optimum MILP cozumu bulunamadi! Lutfen bekleyin veya limitleri esnetin."
        )
        return

    total_risk = 0.0
    total_spent = 0.0
    total_water_cap = 0.0
    assignments = []

    for i in station_ids:
        for j in grid_ids:
            if z[(i, j)].value() == 1:
                assignments.append((i, j, risk_by_grid[j], dist_by_pair[(i, j)]))
                total_risk += risk_by_grid[j]

    for i in station_ids:
        for k in HELI_TYPES:
            count = int(x[(i, k)].value() or 0)
            total_spent += count * HELI_TYPES[k]["cost"]
            total_water_cap += count * HELI_TYPES[k]["water"]

    print("\n=======================================================")
    print("               FILO OPTIMIZASYON SONUCU                ")
    print("=======================================================")
    print(
        f"Toplam Harcanan Butce    : {total_spent} Milyon $ (Limit: {BUDGET} Milyon $)"
    )
    print(f"Filonun Toplam Su Gucu   : {total_water_cap} Galon")
    print(f"Korunan Toplam Risk Skoru: {total_risk:.2f}")
    print("=======================================================")
    print("\nISTASYONLARA ATANAN HELIKOPTERLER:")
    print("-------------------------------------------------------")

    aktif_istasyon_sayisi = 0
    for i in station_ids:
        for k in HELI_TYPES:
            count = int(x[(i, k)].value() or 0)
            if count > 0:
                aktif_istasyon_sayisi += 1
                if k == "Type_1":
                    print(
                        f"Istasyon {i:<3} ---> {k:<8} (BUYUK BOY  - {HELI_TYPES[k]['water']} Galon)"
                    )
                elif k == "Type_2":
                    print(
                        f"Istasyon {i:<3} ---> {k:<8} (ORTA BOY   - {HELI_TYPES[k]['water']} Galon)"
                    )
                elif k == "Type_3":
                    print(
                        f"Istasyon {i:<3} ---> {k:<8} (KUCUK BOY  - {HELI_TYPES[k]['water']} Galon)"
                    )

    print("-------------------------------------------------------")
    print(f"Toplam Aktif Edilen Istasyon Sayisi: {aktif_istasyon_sayisi}")
    print("=======================================================\n")


def main():
    station_ids, grid_ids, risk_by_grid, dist_by_pair, w_j_dict, s_i_dict = load_data()

    top_grids = [j for j in grid_ids if risk_by_grid[j] > 0.0]

    print(
        f"\n{len(station_ids)} Istasyon ve {len(top_grids)} Riskli Bolge ile optimizasyon basliyor..."
    )
    print("Cozucu (Solver) hesapliyor... Lutfen en fazla 60 saniye bekleyin...\n")

    model, x, z = build_milp(station_ids, top_grids, risk_by_grid, dist_by_pair)

    model.solve(PULP_CBC_CMD(msg=False, timeLimit=60))

    summarize_solution(model, x, z, station_ids, top_grids, risk_by_grid, dist_by_pair)
    if model.status == 1:
        analyze_station_decisions(
            model, x, z, station_ids, top_grids, risk_by_grid, dist_by_pair, s_i_dict
        )


if __name__ == "__main__":
    main()
