import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import StringIO

# App configuration
st.set_page_config(layout="wide", page_title="3D Data Visualization & Clustering")

# App title and description
st.title("🌐 3D Data Visualization & Clustering")
st.write("Explore datasets through interactive 3D visualizations and clustering")

# Sidebar for controls
with st.sidebar:
    st.header("Controls & Settings")
    
    # Dataset selection
    dataset = st.selectbox(
        "Choose a dataset:",
        ("Iris", "Gapminder 3D", "Random 3D Clusters", "Custom Upload")
    )
    
    # Clustering options
    st.subheader("Clustering Settings")
    enable_clustering = st.checkbox("Enable Clustering", False)
    if enable_clustering:
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        clustering_algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN"])
    
    # 3D Plot customization
    st.subheader("3D Plot Settings")
    marker_size = st.slider("Marker size", 1, 20, 8)
    opacity = st.slider("Marker opacity", 0.1, 1.0, 0.8)
    theme = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])

# Initialize variables
df = None
fig_3d = None
fig_cluster = None

# Load and process selected dataset
if dataset == "Iris":
    df = px.data.iris()
    with st.expander("Dataset Info"):
        st.write("""
        The Iris dataset contains measurements of iris flowers from three species.
        """)
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        df, x='sepal_length', y='sepal_width', z='petal_length',
        color='species', symbol='species',
        size_max=marker_size, opacity=opacity,
        template=theme,
        title="Iris Dataset 3D Visualization"
    )
    
    if enable_clustering:
        # Prepare data for clustering
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster visualization
        fig_cluster = px.scatter_3d(
            df, x='sepal_length', y='sepal_width', z='petal_length',
            color='cluster', symbol='species',
            size_max=marker_size, opacity=opacity,
            template=theme,
            title="Iris Dataset with Clustering"
        )

elif dataset == "Gapminder 3D":
    df = px.data.gapminder()
    with st.expander("Dataset Info"):
        st.write("""
        Gapminder dataset shows life expectancy, GDP per capita, and population for countries over time.
        """)
    
    # Filter for a specific year for better 3D visualization
    year = st.slider("Select year", min_value=1952, max_value=2007, value=2007, step=5)
    df_year = df[df['year'] == year]
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        df_year, x='gdpPercap', y='lifeExp', z='pop',
        color='continent', size='pop',
        size_max=marker_size*2, opacity=opacity,
        log_x=True, template=theme,
        hover_name='country',
        title=f"Gapminder Data {year} (GDP vs Life Expectancy vs Population)"
    )
    
    if enable_clustering:
        # Prepare data for clustering
        X = df_year[['gdpPercap', 'lifeExp', 'pop']]
        X['pop'] = np.log(X['pop'])  # Log transform population
        X['gdpPercap'] = np.log(X['gdpPercap'])  # Log transform GDP
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_year['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster visualization
        fig_cluster = px.scatter_3d(
            df_year, x='gdpPercap', y='lifeExp', z='pop',
            color='cluster', size='pop',
            size_max=marker_size*2, opacity=opacity,
            log_x=True, template=theme,
            hover_name='country',
            title=f"Gapminder Clusters {year}"
        )

elif dataset == "Random 3D Clusters":
    # Generate synthetic 3D cluster data
    np.random.seed(42)
    n_points = 300
    cluster_1 = np.random.normal(loc=[1, 1, 1], scale=0.3, size=(n_points, 3))
    cluster_2 = np.random.normal(loc=[3, 3, 1], scale=0.4, size=(n_points, 3))
    cluster_3 = np.random.normal(loc=[2, 1, 3], scale=0.2, size=(n_points, 3))
    
    data = np.vstack([cluster_1, cluster_2, cluster_3])
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    df['true_cluster'] = [0]*n_points + [1]*n_points + [2]*n_points
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        df, x='X', y='Y', z='Z',
        color='true_cluster', 
        size_max=marker_size, opacity=opacity,
        template=theme,
        title="Synthetic 3D Clusters (Ground Truth)"
    )
    
    if enable_clustering:
        # Apply clustering
        if clustering_algorithm == "K-Means":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['predicted_cluster'] = kmeans.fit_predict(df[['X', 'Y', 'Z']])
        else:  # DBSCAN
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            df['predicted_cluster'] = dbscan.fit_predict(df[['X', 'Y', 'Z']])
        
        # Create cluster visualization
        fig_cluster = px.scatter_3d(
            df, x='X', y='Y', z='Z',
            color='predicted_cluster', 
            size_max=marker_size, opacity=opacity,
            template=theme,
            title=f"Synthetic 3D Clusters ({clustering_algorithm} Clustering)"
        )

elif dataset == "Custom Upload":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            with st.expander("Data Preview"):
                st.dataframe(df.head())
            
            cols = df.columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis column", cols)
                y_col = st.selectbox("Y-axis column", cols)
            with col2:
                z_col = st.selectbox("Z-axis column", cols)
                color_col = st.selectbox("Color column", [None] + cols)
            
            # Create 3D scatter plot
            fig_3d = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=color_col,
                size_max=marker_size, opacity=opacity,
                template=theme,
                hover_name=df.columns[0]
            )
            
            if enable_clustering:
                # Prepare data for clustering
                X = df[[x_col, y_col, z_col]]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply clustering
                if clustering_algorithm == "K-Means":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df['cluster'] = kmeans.fit_predict(X_scaled)
                else:  # DBSCAN
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps=0.5, min_samples=10)
                    df['cluster'] = dbscan.fit_predict(X_scaled)
                
                # Create cluster visualization
                fig_cluster = px.scatter_3d(
                    df, x=x_col, y=y_col, z=z_col,
                    color='cluster',
                    size_max=marker_size, opacity=opacity,
                    template=theme,
                    hover_name=df.columns[0],
                    title=f"Custom Data Clusters ({clustering_algorithm})"
                )
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Display the 3D plot
if fig_3d is not None:
    st.plotly_chart(fig_3d, use_container_width=True)

# Display the cluster visualization if enabled
if enable_clustering and fig_cluster is not None:
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Show cluster statistics if available
    if 'cluster' in df.columns:
        with st.expander("Cluster Statistics"):
            st.write(df.groupby('cluster').mean())

# Add footer with instructions
st.markdown("---")
st.markdown("""
**Interaction Guide:**
- Rotate: Click and drag
- Zoom: Scroll or use toolbar
- Pan: Right-click and drag
- Reset view: Click the home button in toolbar
- Hover over points for details
""")

# Add download button for data
if df is not None and not enable_clustering:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download current data as CSV",
        data=csv,
        file_name=f"{dataset}_data.csv",
        mime="text/csv"
    )
elif df is not None and enable_clustering:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download clustered data as CSV",
        data=csv,
        file_name=f"{dataset}_clustered_data.csv",
        mime="text/csv"
    )