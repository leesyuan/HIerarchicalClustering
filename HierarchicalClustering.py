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
import seaborn as sns

# Function to plot dendrogram
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, no_labels=True)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Samples')
    plt.ylabel('Distance (Euclidean)')
    st.pyplot(plt)

# Function to plot 2D PCA scatter plot
def plot_pca(X_pca, y_hc):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_hc, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Sales Performance Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

# Function to plot 3D PCA scatter plot
def plot_pca_3d(X_pca, y_hc):
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Cluster'] = y_hc
    fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                        title='Interactive 3D Clusters',
                        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2', 'PCA3': 'PCA Component 3'},
                        opacity=0.8)
    st.plotly_chart(fig)

# Function to plot 2D t-SNE scatter plot
def plot_tsne_2d(X_tsne, y_hc):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_hc, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Clusters Visualized with t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

# Function to plot 3D t-SNE scatter plot
def plot_tsne_3d(X_tsne_3d, y_hc):
    df_tsne = pd.DataFrame(X_tsne_3d, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_tsne['Cluster'] = y_hc
    fig = px.scatter_3d(df_tsne, x='TSNE1', y='TSNE2', z='TSNE3', color='Cluster',
                        title='Interactive 3D t-SNE Visualization',
                        labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2', 'TSNE3': 't-SNE Component 3'},
                        opacity=0.8, width=800, height=600)
    st.plotly_chart(fig)

# Handling missing values and errors in CSV columns
def validate_data(df):
    required_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    
    # Check for missing columns
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the dataset. Please upload a valid CSV file.")
            return False
    
    # Check for NaN or missing values
    if df[required_columns].isnull().any().any():
        st.error("The dataset contains missing values. Please clean your data and try again.")
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
    
    # Check if data is valid
    if validate_data(df):
        try:
            # Selecting features for clustering
            X = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

            # Standardizing the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform hierarchical clustering to generate the linkage matrix
            Z = linkage(X_scaled, method='ward')

            # Plot the dendrogram
            st.subheader('Dendrogram')
            plot_dendrogram(Z)

            # Different linkage methods
            linkage_methods = ['ward', 'complete', 'average', 'single']
            st.subheader('Silhouette Scores for Different Linkage Methods')
            for method in linkage_methods:
                hc = AgglomerativeClustering(n_clusters=2, linkage=method)
                y_hc = hc.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, y_hc)
                st.write(f'For {method} linkage, the Silhouette Score is {silhouette_avg}')

            # Agglomerative Clustering with complete linkage
            hc = AgglomerativeClustering(n_clusters=2, linkage='complete')
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
            plot_pca_3d(X_pca, y_hc)

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
