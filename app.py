import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime as dt

# Set Streamlit Page Config
st.set_page_config(page_title="RFM & Product Recommendation", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('online_retail.csv', encoding='latin1')
    df.dropna(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[~df['InvoiceNo'].str.contains('C', na=False)]
    return df

df = load_data()

# Sidebar Menu
menu = st.sidebar.selectbox("Select Section", ["Exploratory Data Analysis", "RFM Clustering", "Product Recommendation"])

# --- Exploratory Data Analysis ---
if menu == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    # Top 10 Products
    st.subheader("Top 10 Best-Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)
    st.pyplot(fig)

    # Top 10 Customers
    st.subheader("Top 10 Customers by Total Spending")
    top_customers = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_customers.values, y=top_customers.index.astype(str), palette='magma', ax=ax)
    st.pyplot(fig)

    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('InvoiceMonth')['TotalPrice'].sum()
    monthly_sales.index = monthly_sales.index.astype(str)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- RFM Clustering ---
elif menu == "RFM Clustering":
    st.title("📦 RFM Customer Segmentation")
    latest_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (latest_date - date.max()).days,
        'InvoiceNo': lambda num: num.nunique(),
        'TotalPrice': lambda price: price.sum()
    }).reset_index()
    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Elbow Method (optional visualization)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Elbow Method for Optimal K')
    st.pyplot(fig)

    # KMeans with K = 4 (change if needed)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Cluster Summary
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2).reset_index()
    cluster_summary.columns = ['Cluster', 'Recency_Mean', 'Frequency_Mean', 'Monetary_Mean', 'Customer_Count']

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    # 3D Scatter Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for i in range(4):
        cluster_data = rfm_df[rfm_df['Cluster'] == i]
        ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], 
                   color=colors[i], label=f'Cluster {i}', s=50)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Customer Clusters based on RFM')
    ax.legend()
    st.pyplot(fig)

# --- Product Recommendation ---
elif menu == "Product Recommendation":
    st.title("🛒 Product Recommendation Engine")

    # User-Item Matrix
    user_item_matrix = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)

    # Item-Item Cosine Similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.T.index, columns=user_item_matrix.T.index)

    # Product Recommendation Function
    def get_recommendations(product_name, num_recommendations=5):
        if product_name not in item_similarity_df.index:
            return []
        similarity_scores = item_similarity_df[product_name]
        similar_products = similarity_scores.sort_values(ascending=False).head(num_recommendations + 1)
        similar_products = similar_products[similar_products.index != product_name]
        return similar_products.index.tolist()

    # Sample Product Selection
    sample_products = user_item_matrix.columns.to_list()
    selected_product = st.selectbox("Select a Product for Recommendation:", options=sample_products)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_product)
        if recommendations:
            st.success(f"Products similar to '{selected_product}':")
            for prod in recommendations:
                st.write(f"- {prod}")
        else:
            st.error("Product not found in similarity matrix.")
