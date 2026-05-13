#!/usr/bin/env python
# coding: utf-8

# In[4]:


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
scaler = MinMaxScaler()


print("1. Veriler Yükleniyor...")


gdf_pistler = gpd.read_file("raw_data/HelipadLocations.geojson").to_crs(epsg=32635)
gdf_pistler['geometry'] = gdf_pistler.geometry.centroid
gdf_istasyonlar = gpd.GeoDataFrame(
    {'Istasyon_ID': range(len(gdf_pistler))},
    geometry=gdf_pistler['geometry'].values,
    crs="EPSG:32635"
)


gdf_su = gpd.read_file("raw_data/su.geojson", on_invalid="ignore")
gdf_su = gdf_su[gdf_su.geometry.notna()].to_crs(epsg=32635)


df_arc = pd.read_csv("raw_data/fire_archive_SV-C2_739664.csv")
df_nrt = pd.read_csv("raw_data/fire_nrt_SV-C2_739664.csv")

if 'type' in df_arc.columns:
    df_arc = df_arc[(df_arc['type'] == 0) & (df_arc['confidence'].isin(['h', 'n']))]
if 'confidence' in df_nrt.columns:
    df_nrt = df_nrt[df_nrt['confidence'].isin(['h', 'n'])]

df_yangin = pd.concat([df_arc, df_nrt], ignore_index=True)

gdf_yangin = gpd.GeoDataFrame(
    df_yangin,
    geometry=gpd.points_from_xy(df_yangin.longitude, df_yangin.latitude),
    crs="EPSG:4326"
).to_crs(epsg=32635)


gdf_gercek_yanginlar = gdf_yangin.copy()
print(f"Kullanılan yangın sayısı: {len(gdf_gercek_yanginlar)}")



print("\n2. İstanbul Grid'i (Poligon) Oluşturuluyor...")

min_boylam, max_boylam = 28.0, 29.9
min_enlem, max_enlem = 40.8, 41.5
adim = 0.02

grid_poligonlar = []
for x in np.arange(min_boylam, max_boylam, adim):
    for y in np.arange(min_enlem, max_enlem, adim):
        grid_poligonlar.append(box(x, y, x + adim, y + adim))

gdf_grid = gpd.GeoDataFrame(geometry=grid_poligonlar, crs="EPSG:4326").to_crs(epsg=32635)
gdf_grid['Grid_ID'] = range(len(gdf_grid))
print(f"Toplam grid sayısı: {len(gdf_grid)}")



print("\n3. w_j (Su Uzaklığı) Hesaplanıyor...")

grid_su_join = gpd.sjoin_nearest(gdf_grid, gdf_su, how='left', distance_col='w_j')
w_j_df = grid_su_join.groupby('Grid_ID')['w_j'].min().reset_index()



print("\n4. Kompozit R_j Hesaplanıyor...")


gdf_yangin_temiz = gdf_gercek_yanginlar.drop(
    columns=[c for c in gdf_gercek_yanginlar.columns if 'index_' in c],
    errors='ignore'
).reset_index(drop=True)

gdf_yangin_temiz['acq_date'] = pd.to_datetime(gdf_yangin_temiz['acq_date'])
gdf_yangin_temiz['ay'] = gdf_yangin_temiz['acq_date'].dt.month
gdf_yangin_temiz['mevsim_agirlik'] = gdf_yangin_temiz['ay'].apply(
    lambda m: 1.0 if m in [6, 7, 8, 9] else 0.3
)
gdf_yangin_temiz['gece'] = (gdf_yangin_temiz['daynight'] == 'N').astype(int)


join_yangin = gpd.sjoin(
    gdf_yangin_temiz,
    gdf_grid[['Grid_ID', 'geometry']],
    how='inner',
    predicate='intersects'
)
print(f"Eşleşen yangın-grid çifti: {len(join_yangin)}")


risk_stats = join_yangin.groupby('Grid_ID').agg(
    Frekans=('acq_date', 'count'),
    Ort_FRP=('frp', 'mean'),
    Ort_Mevsim=('mevsim_agirlik', 'mean'),
    Gece_Orani=('gece', 'mean')
).reset_index()


for col in ['Frekans', 'Ort_FRP', 'Ort_Mevsim', 'Gece_Orani']:
    risk_stats[f'{col}_norm'] = scaler.fit_transform(risk_stats[[col]])


risk_stats = risk_stats.merge(w_j_df, on='Grid_ID', how='left')
risk_stats['w_j_norm'] = scaler.fit_transform(risk_stats[['w_j']])


risk_stats['Ham_Risk'] = (
    0.35 * risk_stats['Frekans_norm'] +
    0.25 * risk_stats['Ort_FRP_norm'] +
    0.20 * risk_stats['Ort_Mevsim_norm'] +
    0.10 * risk_stats['Gece_Orani_norm'] +
    0.10 * risk_stats['w_j_norm']
)

gdf_grid = gdf_grid.merge(
    risk_stats[['Grid_ID', 'Ham_Risk']],
    on='Grid_ID', how='left'
).fillna(0)

gdf_grid['R_j'] = scaler.fit_transform(gdf_grid[['Ham_Risk']])

print(f"Risk > 0 grid sayısı: {(gdf_grid['R_j'] > 0).sum()}")
print(f"Max R_j: {gdf_grid['R_j'].max():.4f}")

df_Rj = gdf_grid[['Grid_ID', 'R_j']]
df_Rj.to_csv("Rj_Risk_Matrisi.csv", index=False)
print("--> Rj_Risk_Matrisi.csv kaydedildi!")



print("\n5. D_i,j Mesafe Matrisi Hesaplanıyor...")


istasyon_su_join = gpd.sjoin_nearest(gdf_istasyonlar, gdf_su, how='left', distance_col='s_i')
s_i_df = istasyon_su_join.groupby('Istasyon_ID')['s_i'].min().reset_index()

alpha = 5  

mesafe_matrisi = []
for i_row in s_i_df.itertuples():
    for j_row in w_j_df.itertuples():
        D_ij_km = (i_row.s_i + (alpha + 1) * j_row.w_j) / 1000.0
        mesafe_matrisi.append({
            'Istasyon_ID': i_row.Istasyon_ID,
            'Grid_ID': j_row.Grid_ID,
            'D_ij_km': D_ij_km
        })

df_Dij = pd.DataFrame(mesafe_matrisi)
df_Dij.to_csv("Dij_Mesafe_Matrisi.csv", index=False)
print("--> Dij_Mesafe_Matrisi.csv kaydedildi!")



gdf_grid_export = gdf_grid[['Grid_ID']].copy()
gdf_grid_export['lon'] = gdf_grid.geometry.centroid.to_crs(epsg=4326).x
gdf_grid_export['lat'] = gdf_grid.geometry.centroid.to_crs(epsg=4326).y
gdf_grid_export.to_csv("Grid_Coordinates.csv", index=False)
print("--> Grid_Coordinates.csv kaydedildi!")

print("\nTüm işlemler tamamlandı!")
print(f"  İstasyon sayısı : {len(gdf_istasyonlar)}")
print(f"  Grid sayısı     : {len(gdf_grid)}")
print(f"  D_ij satır sayısı: {len(df_Dij)}")


# In[3]:


import os


output_dir = "cleaned_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"6. Temizlenmiş Veriler '{output_dir}' klasörüne kaydediliyor...")


gdf_yangin_temiz.to_file(f"{output_dir}/fire_clean.geojson", driver='GeoJSON')


gdf_su.to_file(f"{output_dir}/water_clean.geojson", driver='GeoJSON')


gdf_istasyonlar.to_file(f"{output_dir}/helipads_clean.geojson", driver='GeoJSON')

print("--> Tüm temizlenmiş dosyalar başarıyla dışa aktarıldı:")
print(f"    - {output_dir}/fire_clean.geojson")
print(f"    - {output_dir}/water_clean.geojson")
print(f"    - {output_dir}/helipads_clean.geojson")


# In[10]:


df_Rj = pd.read_csv("Rj_Risk_Matrisi.csv")


riskli_gridler = df_Rj[df_Rj['R_j'] > 0]

print(f"Risk taşıyan toplam ormanlık/riskli bölge sayısı: {len(riskli_gridler)}")


if 'gdf_grid' in locals():
    fig, ax = plt.subplots(figsize=(15, 8))


    gdf_grid.plot(ax=ax, color='lightgrey', alpha=0.2)


    gdf_grid_riskli = gdf_grid[gdf_grid['R_j'] > 0]
    gdf_grid_riskli.plot(ax=ax, column='R_j', cmap='YlOrRd', legend=True, 
                         markersize=15, alpha=0.8)

    plt.title("İstanbul Orman Yangını Risk Isı Haritası (R_j > 0)")
    plt.show()

