import streamlit as st
import pandas as pd
import folium
from math import radians, sin, cos, sqrt, atan2
from streamlit_folium import st_folium
import heapq  
from folium.features import DivIcon

# Fungsi untuk menghitung jarak haversine (dalam kilometer)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

data = pd.read_csv('data.csv')

st.sidebar.header("Filter Data")
province_name = st.sidebar.selectbox(
    "Pilih Provinsi",
    options=data['province_name'].unique(),
    index=0)

stage = st.sidebar.selectbox(
    "Pilih Tahapan Pendidikan",
    options=data['stage'].unique(),
    index=0)

city_name = st.sidebar.selectbox(
    "Pilih Kota",
    options=data[data['province_name'] == province_name]['city_name'].unique(),
    index=0)

tempat = data[
    (data['province_name'] == province_name) & 
    (data['stage'] == stage) & 
    (data['city_name'] == city_name)
]

tempat_clean = tempat.dropna(subset=['lat', 'long'])
tempat_clean = tempat_clean[(tempat_clean['lat'] != 0.0) & (tempat_clean['long'] != 0.0)]
# Visualisasi pertama: Semua koneksi antar lokasi dengan CircleMarker
st.subheader("Visualisasi 1: Semua Koneksi Antar Lokasi")
if not tempat_clean.empty:
    m1 = folium.Map(
        location=[tempat_clean['lat'].mean(), tempat_clean['long'].mean()],
        zoom_start=10
    )

    # Tambahkan CircleMarker untuk setiap lokasi
    for i, row in tempat_clean.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,  # Ukuran titik
            color="blue",  # Warna lingkaran
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup=f"{row['school_name']} ({row['district']})",
        ).add_to(m1)

    # Tambahkan koneksi antar lokasi
    coordinates = tempat_clean[['lat', 'long']].values.tolist()
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            folium.PolyLine(
                locations=[coordinates[i], coordinates[j]],
                color="blue",
                weight=2.5,
                opacity=1
            ).add_to(m1)

    # Render peta di Streamlit
    st_folium(m1, width=700, height=500)
else:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih untuk Visualisasi 1.")




@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


@st.cache_data
def calculate_edges(tempat_clean):
    nodes = list(tempat_clean.index)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            lat1, lon1 = tempat_clean.loc[node1, ["lat", "long"]]
            lat2, lon2 = tempat_clean.loc[node2, ["lat", "long"]]
            distance = haversine(lat1, lon1, lat2, lon2)
            edges.append((distance, node1, node2))
    return sorted(edges, key=lambda x: x[0])


# Fungsi algoritma Kruskal
@st.cache_data
def kruskal_mst(nodes, edges):
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    mst_edges = []
    for weight, node1, node2 in edges:
        if find(node1) != find(node2):
            union(node1, node2)
            mst_edges.append((node1, node2, weight))
    return mst_edges


# Fungsi algoritma Prim
@st.cache_data
def prim_mst(nodes, edges):
    start_node = nodes[0]
    mst_edges = []
    visited = set()
    edge_heap = []

    def add_edges(node):
        visited.add(node)
        for weight, node1, node2 in edges:
            if node1 == node and node2 not in visited:
                heapq.heappush(edge_heap, (weight, node1, node2))
            elif node2 == node and node1 not in visited:
                heapq.heappush(edge_heap, (weight, node2, node1))

    add_edges(start_node)
    while len(visited) < len(nodes) and edge_heap:
        weight, node1, node2 = heapq.heappop(edge_heap)
        if node2 not in visited:
            mst_edges.append((node1, node2, weight))
            add_edges(node2)
    return mst_edges


# Fungsi algoritma Boruvka
@st.cache_data
def boruvka_mst(nodes, edges):
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    mst_edges = []
    while len(mst_edges) < len(nodes) - 1:
        cheapest = {node: None for node in nodes}
        for weight, node1, node2 in edges:
            root1, root2 = find(node1), find(node2)
            if root1 != root2:
                if cheapest[root1] is None or weight < cheapest[root1][0]:
                    cheapest[root1] = (weight, node1, node2)
                if cheapest[root2] is None or weight < cheapest[root2][0]:
                    cheapest[root2] = (weight, node1, node2)

        for node in nodes:
            if cheapest[node] is not None:
                weight, node1, node2 = cheapest[node]
                if find(node1) != find(node2):
                    union(node1, node2)
                    mst_edges.append((node1, node2, weight))
    return mst_edges


# Fungsi untuk membuat peta Folium
def create_map(tempat_clean, mst_edges, title):
    m = folium.Map(
        location=[tempat_clean["lat"].mean(), tempat_clean["long"].mean()],
        zoom_start=10
    )
    for i, row in tempat_clean.iterrows():
        folium.Marker(
            location=[row["lat"], row["long"]],
            popup=f"{row['school_name']} ({row['district']})",
        ).add_to(m)

    for node1, node2, weight in mst_edges:
        lat1, lon1 = tempat_clean.loc[node1, ["lat", "long"]]
        lat2, lon2 = tempat_clean.loc[node2, ["lat", "long"]]
        mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        folium.PolyLine([[lat1, lon1], [lat2, lon2]], color="blue", weight=2.5).add_to(m)
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 10pt; color: black;">{weight:.2f} km</div>',
            ),
        ).add_to(m)
    return m


if not tempat_clean.empty:
    nodes = list(tempat_clean.index)
    edges = calculate_edges(tempat_clean)

    st.subheader("Visualisasi 2: Kruskal MST")
    mst_edges_kruskal = kruskal_mst(nodes, edges)
    st_folium(create_map(tempat_clean, mst_edges_kruskal, "Kruskal MST"), width=700)

    st.subheader("Visualisasi 3: Prim MST")
    mst_edges_prim = prim_mst(nodes, edges)
    st_folium(create_map(tempat_clean, mst_edges_prim, "Prim MST"), width=700)

    st.subheader("Visualisasi 4: Boruvka MST")
    mst_edges_boruvka = boruvka_mst(nodes, edges)
    st_folium(create_map(tempat_clean, mst_edges_boruvka, "Boruvka MST"), width=700)
else:
    st.warning("Tidak ada data yang sesuai dengan filter.")
