import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ── Helper ────────────────────────────────────────────────────
def set_k(val: int):
    st.session_state["k_value"] = int(val)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Annual Income (k$)'    : 'Annual Income',
        'Spending Score (1-100)': 'Spending Score',
        'CustomerID'            : 'Customer ID'
    })
    return df

df = load_data()

# ── Title ─────────────────────────────────────────────────────
st.title("🛍️ Mall Customer Segmentation")
st.caption("Unsupervised Learning — K-Means Clustering")
st.markdown("---")

# ── Dataset Overview ──────────────────────────────────────────
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Age Range", f"{df['Age'].min()}–{df['Age'].max()}")
col3.metric("Income Range", f"{df['Annual Income'].min()}k–{df['Annual Income'].max()}k")
col4.metric("Spending Score Range", f"{df['Spending Score'].min()}–{df['Spending Score'].max()}")

st.dataframe(df.head(10), use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Settings")

if "k_value" not in st.session_state:
    st.session_state["k_value"] = 3

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    step=1,
    key="k_value"
)

# ── Features ──────────────────────────────────────────────────
X = df[["Annual Income", "Spending Score"]].values

# ── Elbow Method ──────────────────────────────────────────────
st.markdown("---")
st.header("Elbow Method — Choosing K")

ks       = list(range(2, 11))
inertias = []
for kk in ks:
    km_tmp = KMeans(n_clusters=kk, random_state=42, n_init="auto")
    km_tmp.fit(X)
    inertias.append(km_tmp.inertia_)

knee      = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
optimal_k = knee.knee

col_left, col_right = st.columns([2, 1])

with col_left:
    fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=120)
    ax1.plot(ks, inertias, marker="o", color="steelblue", linewidth=2, markersize=8)
    ax1.axvline(st.session_state["k_value"], color="orange", linestyle="--",
                linewidth=2, label=f"Selected K = {st.session_state['k_value']}")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Curve", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with col_right:
    st.markdown("### Suggested K")
    if optimal_k is not None:
        st.metric("Optimal K (KneeLocator)", int(optimal_k))
        st.button("Use suggested K", on_click=set_k,
                  args=(int(optimal_k),), key="use_optimal_k_btn")
    else:
        st.warning("No clear elbow found.")

# ── Run K-Means ───────────────────────────────────────────────
st.markdown("---")
st.header(f"K-Means Clustering — K = {st.session_state['k_value']}")

km     = KMeans(n_clusters=int(st.session_state["k_value"]),
                random_state=42, n_init="auto")
labels = km.fit_predict(X)

df_out            = df.copy()
df_out["Cluster"] = labels
df_out["Cluster"] = df_out["Cluster"].apply(lambda x: f"Cluster {x+1}")

centers = km.cluster_centers_

# ── Scatter Plot ──────────────────────────────────────────────
st.subheader("Cluster Visualization")

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    palette = sns.color_palette("Set2", st.session_state["k_value"])
    for i, cluster in enumerate(sorted(df_out["Cluster"].unique())):
        subset = df_out[df_out["Cluster"] == cluster]
        ax.scatter(subset["Annual Income"], subset["Spending Score"],
                   label=cluster, color=palette[i], s=60, alpha=0.85,
                   edgecolors="white", linewidth=0.5)
    ax.scatter(centers[:, 0], centers[:, 1],
               marker="*", s=300, color="black", label="Centroid", zorder=5)
    ax.set_xlabel("Annual Income (k$)", fontsize=12)
    ax.set_ylabel("Spending Score", fontsize=12)
    ax.set_title("Annual Income vs Spending Score", fontsize=14)
    ax.legend(title="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_info:
    st.markdown("### Centroids")
    cent_df = pd.DataFrame(centers,
                           columns=["Annual Income", "Spending Score"]).round(2)
    cent_df.index = [f"Cluster {i+1}" for i in range(st.session_state["k_value"])]
    st.dataframe(cent_df, use_container_width=True)

    st.markdown("### Cluster Counts")
    counts = df_out["Cluster"].value_counts().sort_index().to_frame("Count")
    st.dataframe(counts, use_container_width=True)

# ── Cluster Profiles ──────────────────────────────────────────
st.markdown("---")
st.header("Cluster Profiles")
st.caption("Mean values per cluster — use this to interpret what each cluster represents.")

profile = df_out[["Age", "Annual Income", "Spending Score", "Cluster"]]\
    .groupby("Cluster").mean().round(1)
profile.insert(0, "Count", df_out.groupby("Cluster").size())
st.dataframe(profile, use_container_width=True)

# ── Download ──────────────────────────────────────────────────
st.markdown("---")
csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Clustered Data", csv,
                   "clustered_customers.csv", "text/csv")
