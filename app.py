import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("Analisis Risiko Bencana di Sumatera Barat (2019–2021)")

# Upload file
data_file = st.file_uploader("Upload file CSV Data Bencana:", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)

    # Bersihkan dan siapkan data
    df = df[df['Kabupaten/Kota'].str.strip() != 'Provinsi Sumatera Barat']
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    df = df[df['Kabupaten_Kota'] != 'Provinsi Sumatera Barat']

    numerical_columns = [
        'Banjir_2019', 'Banjir_2020', 'Banjir_2021',
        'Gempa_2019', 'Gempa_2020', 'Gempa_2021',
        'Longsor_2019', 'Longsor_2020', 'Longsor_2021'
    ]

    df[numerical_columns] = df[numerical_columns].replace('-', 0).astype(float)
    df.drop_duplicates(inplace=True)

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numerical_columns])

    # Tentukan jumlah cluster optimal
    wcss = []
    silhouette_scores = []
    range_k = range(2, 7)

    for k in range_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    st.subheader("Pemilihan Jumlah Cluster Optimal")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range_k, wcss, marker='o')
    ax[0].set_title('Metode Elbow')
    ax[0].set_xlabel('Jumlah Cluster (k)')
    ax[0].set_ylabel('WCSS')

    ax[1].plot(range_k, silhouette_scores, marker='o', color='green')
    ax[1].set_title('Silhouette Score')
    ax[1].set_xlabel('Jumlah Cluster (k)')
    ax[1].set_ylabel('Skor Silhouette')
    st.pyplot(fig)

    optimal_k = range_k[np.argmax(silhouette_scores)]
    st.success(f"Jumlah cluster optimal berdasarkan Silhouette Score: {optimal_k}")

    # KMeans Clustering
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

    # Rata-rata bencana per cluster
    cluster_means = df.groupby('Cluster_KMeans')[numerical_columns].mean()
    cluster_means['Total_Bencana'] = cluster_means.sum(axis=1)
    st.subheader("Rata-rata Jumlah Bencana per Cluster")
    st.dataframe(cluster_means)

    # Mapping risiko
    sorted_clusters = cluster_means['Total_Bencana'].sort_values(ascending=False)
    cluster_labels = {}
    labels = ['Risiko Tinggi', 'Risiko Sedang', 'Risiko Rendah']
    for i, cluster_id in enumerate(sorted_clusters.index):
        cluster_labels[cluster_id] = labels[i] if i < len(labels) else f"Risiko Lain {i}"

    df['Label_Risiko'] = df['Cluster_KMeans'].map(cluster_labels)

    # Visualisasi PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.subheader("Visualisasi Clustering dengan PCA")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('Set2', df['Cluster_KMeans'].nunique())

    for i in range(df.shape[0]):
        ax2.text(X_pca[i, 0], X_pca[i, 1], df.iloc[i]['Kabupaten_Kota'], fontsize=8, alpha=0.6)

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Label_Risiko'], palette=palette, s=100, edgecolor='black', ax=ax2)
    ax2.set_title("Clustering KMeans dengan PCA + Label Risiko")
    ax2.set_xlabel("PCA Komponen 1")
    ax2.set_ylabel("PCA Komponen 2")
    ax2.grid(True)
    st.pyplot(fig2)

    # Daftar daerah risiko tinggi
    st.subheader("Daerah dengan Risiko Tinggi")
    rawan_tinggi = df[df['Label_Risiko'] == 'Risiko Tinggi']
    st.dataframe(rawan_tinggi[['Kabupaten_Kota', 'Label_Risiko']])

    # Korelasi antar bencana
    st.subheader("Korelasi Antar Jenis Bencana")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
