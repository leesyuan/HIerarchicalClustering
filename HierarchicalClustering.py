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

# Handling column errors in CSV columns
def validate_columns(df):
    required_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    
    # Check for missing columns
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the dataset. Please upload a valid CSV file.")
            return False
    return True

# Function to plot average sales per cluster
def plot_avg_sales_per_cluster(df):
    numerical_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
    
    # Group the data by 'Cluster' and calculate the mean of each numerical column
    cluster_means = df.groupby('Cluster')[numerical_columns].mean()
    
    # Select sales columns for the bar graph
    sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    
    # Create the bar graph
    cluster_means[sales_columns].plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.legend(title='Sales Region')
    st.pyplot(plt)

# Function to plot genre distribution per cluster
def plot_genre_distribution(df):
    # Group the data by cluster and genre, then count the occurrences of each genre within each cluster
    genre_counts = df.groupby(['Cluster', 'Genre'])['Genre'].count().unstack()
    
    # Create the bar chart
    genre_counts.plot(kind='bar', figsize=(12, 6))
    plt.title('Genre Distribution per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=0)
    plt.legend(title='Genre')
    st.pyplot(plt)

# Function to plot platform distribution per cluster
def plot_platform_distribution(df):
    # Group the data by cluster and platform, then count the occurrences of each platform within each cluster
    platform_counts = df.groupby(['Cluster', 'Platform'])['Platform'].count().unstack()

    # Create the bar chart
    platform_counts.plot(kind='bar', figsize=(15, 6))
    plt.title('Platform Distribution per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=0)
    plt.legend(title='Platform')
    st.pyplot(plt)

# Function to plot top publishers pie charts for each cluster
def plot_top_publishers(df):
    # Group the data by cluster and publisher, then count the occurrences of each publisher within each cluster
    publisher_counts = df.groupby(['Cluster', 'Publisher'])['Publisher'].count().unstack()

    # Select the top 10 publishers for each cluster and handle NaN values by filling them with 0
    top_publishers = publisher_counts.apply(lambda x: x.nlargest(10), axis=1).fillna(0)

    # Iterate through each cluster to plot the pie charts
    for cluster in top_publishers.index:
        # Get the publisher counts for the current cluster
        cluster_counts = top_publishers.loc[cluster]

        # Filter out publishers with zero counts (to avoid empty pie slices)
        cluster_counts = cluster_counts[cluster_counts > 0]

        # Create the pie chart
        plt.figure(figsize=(6, 4))
        plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Top 10 Publishers in Cluster {cluster}')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(plt)
        st.write()
        
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
