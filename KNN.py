import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import messagebox

dataset = pd.read_csv('user+data.csv')

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialiser et entraîner le classificateur KNN
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

root = tk.Tk()
root.title("KNN Classifier - Interface Graphique")

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Cliquez pour afficher les résultats", font=("Arial", 12))
label.pack()

def show_results():
    result_text = f"Matrice de Confusion :\n{cm}"
    messagebox.showinfo("Résultats", result_text)

def plot_decision_boundary():
    fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('K-NN Frontière de Décision')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

button1 = tk.Button(frame, text="Afficher Matrice de Confusion", command=show_results, font=("Arial", 12), bg="blue", fg="white")
button1.pack(pady=5)

button2 = tk.Button(frame, text="Tracer Frontière de Décision", command=plot_decision_boundary, font=("Arial", 12), bg="green", fg="white")
button2.pack(pady=5)

root.mainloop()
