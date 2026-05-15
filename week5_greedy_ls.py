import pandas as pd
import geopandas as gpd
import time as tm

# --- PARAMETRELER (Onceki haftalarla uyumlu) ---
BUDGET = 311
MAX_ASSIGNMENTS_PER_HELI = 20
MAX_STATION_CAPACITY = 1
DISTANCE_LIMIT_KM = 150.0

HELI_TYPES = {
    1: {"name": "Type_1", "cost": 30, "water": 700},
    2: {"name": "Type_2", "cost": 10, "water": 300},
    3: {"name": "Type_3", "cost": 4, "water": 180},
}


# --- VERI YUKLEME ---
def load_data():
    df_risk = pd.read_csv("Rj_Risk_Matrisi.csv")
    grid_ids = sorted(df_risk["Grid_ID"].unique())
    risk_by_grid = df_risk.set_index("Grid_ID")["R_j"].to_dict()

    gdf_stations = gpd.read_file("helipads_clean.geojson", on_invalid="ignore")
    station_ids = sorted(gdf_stations["Istasyon_ID"].unique())

    gdf_su = gpd.read_file("su.geojson", on_invalid="ignore").to_crs(epsg=32635)
    gdf_stations = gdf_stations.to_crs(epsg=32635)

    df_grid = pd.read_csv("Grid_Coordinates.csv")
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
    for i_row in s_i_df.itertuples():
        for j_row in w_j_df.itertuples():
            dist_by_pair[(i_row.Istasyon_ID, j_row.Grid_ID)] = (
                i_row.s_i + (alpha + 1) * j_row.w_j
            ) / 1000.0

    return station_ids, grid_ids, risk_by_grid, dist_by_pair


# --- WEEK 5: GREEDY + LOCAL SEARCH ALGORITMASI ---
def solve_greedy_local_search(station_ids, grid_ids, risk_by_grid, dist_by_pair):
    start_time = tm.perf_counter()

    # 1. GRIDLERI RISKE GORE SIRALA (Prioritize high-R_j regions)
    active_grids = [j for j in grid_ids if risk_by_grid.get(j, 0) > 0]
    sorted_grids = sorted(active_grids, key=lambda j: risk_by_grid[j], reverse=True)

    # Durum Degiskenleri
    current_budget = BUDGET
    active_stations = {}  # istasyon_id -> helikopter tipi (1, 2, 3)
    grid_assignments = {}  # grid_id -> istasyon_id
    station_loads = {i: 0 for i in station_ids}

    # ==========================================
    # ASAMA 1: GREEDY (ACGOZLU) ATAMA
    # ==========================================
    for j in sorted_grids:
        # Bu grid'e erisebilen istasyonlari mesafe (D_ij) acisindan sirala
        feasible_stations = []
        for i in station_ids:
            d = dist_by_pair.get((i, j), float("inf"))
            if d <= DISTANCE_LIMIT_KM:
                feasible_stations.append((d, i))

        # Assign to min D_{i,j} feasible station
        feasible_stations.sort(key=lambda x: x[0])

        for d, i in feasible_stations:
            # Eger istasyon zaten aciksa ve kapasitesi varsa direkt ata
            if i in active_stations:
                if station_loads[i] < MAX_ASSIGNMENTS_PER_HELI:
                    grid_assignments[j] = i
                    station_loads[i] += 1
                    break
            # Eger istasyon kapaliysa ve butce varsa, en ucuz heli ile ac ve ata
            else:
                cheapest_cost = HELI_TYPES[3]["cost"]
                if current_budget >= cheapest_cost:
                    active_stations[i] = 3  # Type 3 koy
                    current_budget -= cheapest_cost
                    grid_assignments[j] = i
                    station_loads[i] += 1
                    break

    # ==========================================
    # ASAMA 2: LOCAL SEARCH (IYLESTIRME)
    # ==========================================

    # Local Search 1: Artan Butceyle Helikopterleri Yukselt (Upgrade)
    for i in list(active_stations.keys()):
        current_type = active_stations[i]
        # Type 1'i (en buyuk) veya Type 2'yi (orta) deneyecegiz
        for target_type in [1, 2]:
            if current_type > target_type:  # Sadece daha buyuge gecis
                upgrade_cost = (
                    HELI_TYPES[target_type]["cost"] - HELI_TYPES[current_type]["cost"]
                )
                if current_budget >= upgrade_cost:
                    active_stations[i] = target_type
                    current_budget -= upgrade_cost
                    break

    # Local Search 2: Gridleri Daha Yakin Istasyonlara Kaydirma (Swap Pass)
    improved = True
    while improved:
        improved = False
        for j in list(grid_assignments.keys()):
            current_i = grid_assignments[j]
            current_d = dist_by_pair.get((current_i, j), float("inf"))

            best_new_i = None
            best_new_d = current_d

            for i in active_stations:
                if i != current_i and station_loads[i] < MAX_ASSIGNMENTS_PER_HELI:
                    new_d = dist_by_pair.get((i, j), float("inf"))
                    if new_d < best_new_d:
                        best_new_d = new_d
                        best_new_i = i

            if best_new_i is not None:
                grid_assignments[j] = best_new_i
                station_loads[current_i] -= 1
                station_loads[best_new_i] += 1
                improved = True

    end_time = tm.perf_counter()

    # Sonuclari Hesapla
    total_dist_risk = 0.0
    for j, i in grid_assignments.items():
        total_dist_risk += risk_by_grid[j] * dist_by_pair[(i, j)]

    unreachable_risk = sum(
        risk_by_grid[j] for j in sorted_grids if j not in grid_assignments
    )
    total_water = sum(HELI_TYPES[active_stations[i]]["water"] for i in active_stations)
    total_spent = BUDGET - current_budget

    objective_score = (
        total_dist_risk + (unreachable_risk * 1000) - (0.001 * total_water)
    )

    return {
        "time": end_time - start_time,
        "score": objective_score,
        "spent": total_spent,
        "water": total_water,
        "active_stations": active_stations,
        "assigned_grids": len(grid_assignments),
        "unreachable_grids": len(sorted_grids) - len(grid_assignments),
        "dist_risk": total_dist_risk,
    }


if __name__ == "__main__":
    print("Veriler yukleniyor...")
    st_ids, gr_ids, r_dict, d_pair = load_data()

    print("\nWeek 5: Greedy + Local Search Calistiriliyor...")
    res = solve_greedy_local_search(st_ids, gr_ids, r_dict, d_pair)

    print("\n=======================================================")
    print("         WEEK 5: GREEDY + LOCAL SEARCH SONUCU          ")
    print("=======================================================")
    print(f"Hesaplama Suresi       : {res['time']:.5f} saniye")
    print(f"Toplam Harcanan Butce  : {res['spent']} Milyon $ (Limit: 311)")
    print(f"Toplam Su Kapasitesi   : {res['water']} Galon")
    print(f"Aktif Istasyon Sayisi  : {len(res['active_stations'])}")
    print(f"Atanan Riskli Grid     : {res['assigned_grids']}")
    print(f"En Iyi Objective Skor  : {res['score']:.2f}")
    print("-------------------------------------------------------")

    print("Envanter Ozeti:")
    type_counts = {"Type_1": 0, "Type_2": 0, "Type_3": 0}
    for h in res["active_stations"].values():
        type_counts[HELI_TYPES[h]["name"]] += 1

    print(f"  Type_1 (Buyuk) : {type_counts['Type_1']} Adet")
    print(f"  Type_2 (Orta)  : {type_counts['Type_2']} Adet")
    print(f"  Type_3 (Kucuk) : {type_counts['Type_3']} Adet")

    print("\n[KIYASLAMA NOTU]")
    print("Bu metot Genetik Algoritmadan ve MILP'den cok daha hizlidir.")
    print("Ancak buldugu cozum Global Minimum olmayabilir (Local Minimumda kalabilir).")
