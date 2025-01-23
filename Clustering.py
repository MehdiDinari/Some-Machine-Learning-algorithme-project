import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Generate synthetic dataset
X, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)

# Create the main application window
root = tk.Tk()
root.title("KMeans Clustering App")
root.geometry("900x700")
root.configure(bg="#f0f0f0")

# Styled frame
frame = tk.Frame(root, bg="white", relief=tk.RAISED, borderwidth=5)
frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Global variable to track the mode (2D or 3D)
view_mode = "2D"
canvas = None


def run_kmeans():
    global canvas, view_mode
    try:
        n_clusters = int(cluster_entry.get())
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        centers = kmeans.cluster_centers_

        # Clear previous canvas
        for widget in frame.winfo_children():
            widget.destroy()

        if view_mode == "2D":
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
            ax.set_title("KMeans Clustering (2D)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend()
        else:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=200, label='Centroids')
            ax.set_title("KMeans Clustering (3D)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    except ValueError:
        result_label.config(text="Invalid Input! Please enter a valid number of clusters.", fg="red")


def toggle_view():
    global view_mode
    view_mode = "3D" if view_mode == "2D" else "2D"
    run_kmeans()


# UI Components
cluster_label = tk.Label(root, text="Enter Number of Clusters:", font=("Arial", 12, "bold"), bg="#f0f0f0")
cluster_label.pack()

cluster_entry = tk.Entry(root, font=("Arial", 12))
cluster_entry.pack(pady=5)

run_button = tk.Button(root, text="Run KMeans", command=run_kmeans, font=("Arial", 12, "bold"), bg="#007bff",
                       fg="white", padx=10, pady=5)
run_button.pack(pady=10)

toggle_button = tk.Button(root, text="Switch to 3D",
                          command=lambda: [toggle_view(), toggle_button.config(text=f"Switch to {view_mode}")],
                          font=("Arial", 12, "bold"), bg="#28a745", fg="white", padx=10, pady=5)
toggle_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f0f0f0")
result_label.pack()

# Run Tkinter Main Loop
root.mainloop()
