import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE

# Function to plot dendrogram
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, no_labels=True)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Samples')
    plt.ylabel('Distance (Euclidean)')
    st.pyplot(plt)

# Other plotting functions as defined before (plot_pca, plot_tsne_2d, plot_tsne_3d, etc.)

# Handling column errors in CSV columns
def validate_columns(df):
    required_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    
    # Check for missing columns
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the dataset. Please upload a valid CSV file.")
            return False
    return True

# Streamlit app layout
st.title('Hierarchical Clustering Analysis of Video Games Sales Data')

# Add a file uploader to allow users to upload CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("Here are the first few rows of your file:")
    st.write(df.head())

    # Preprocessing Step 1: Replace 0s in 'Year' column with NaN
    df['Year'] = df['Year'].replace(0, np.nan)

    # Preprocessing Step 2: Forward-fill the missing values in 'Year' column
    df['Year'] = df['Year'].ffill()

    # Preprocessing Step 3: Replace NaN values in 'Publisher' column with 'Unknown'
    if 'Publisher' in df.columns:
        df['Publisher'] = df['Publisher'].fillna('Unknown')

    # Check if data has required columns
    if validate_columns(df):
        try:
            # User Input: Number of Clusters and Linkage Method
            num_clusters = st.slider('Select Number of Clusters:', min_value=2, max_value=10, value=2)
            linkage_method = st.selectbox('Select Linkage Method:', ('ward', 'complete', 'average', 'single'))

            # Selecting features for clustering
            X = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

            # Standardizing the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform hierarchical clustering to generate the linkage matrix
            Z = linkage(X_scaled, method=linkage_method)

            # Plot the dendrogram
            st.subheader('Dendrogram')
            plot_dendrogram(Z)

            # Perform Agglomerative Clustering
            hc = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
            y_hc = hc.fit_predict(X_scaled)

            # Metrics
            silhouette_avg = silhouette_score(X_scaled, y_hc)
            db_score = davies_bouldin_score(X_scaled, y_hc)
            st.write(f'Silhouette Score: {silhouette_avg}')
            st.write(f'Davies-Bouldin Score: {db_score}')

            # PCA and t-SNE visualizations
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            st.subheader('PCA Visualization')
            plot_pca(X_pca, y_hc)

            tsne = TSNE(n_components=3, random_state=42)
            X_tsne_3d = tsne.fit_transform(X_scaled)
            st.subheader('t-SNE Visualization')
            plot_tsne_2d(X_tsne_3d, y_hc)
            plot_tsne_3d(X_tsne_3d, y_hc)

            # Analysis
            df['Cluster'] = y_hc
            st.subheader('Cluster Analysis')

            # Example plots for further analysis
            plot_avg_sales_per_cluster(df)
            plot_genre_distribution(df)
            plot_platform_distribution(df)
            plot_top_publishers(df)

        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
