import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
from ttkbootstrap.constants import *

dataset = pd.read_csv("user+data.csv")

X = dataset.iloc[:, [2, 4]].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


def show_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)

    cm_window = tk.Toplevel(root)
    cm_window.title("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    canvas = FigureCanvasTkAgg(fig, master=cm_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


root = tb.Window(themename="superhero")
root.title("Logistic Regression GUI")
root.geometry("400x300")

label = tb.Label(root, text="Logistic Regression Model", font=("Arial", 16, "bold"))
label.pack(pady=10)

process_button = tb.Button(root, text="Process Data", bootstyle=SUCCESS, command=lambda: print("Data Processed!"))
process_button.pack(pady=10)

cm_button = tb.Button(root, text="Show Confusion Matrix", bootstyle=PRIMARY, command=show_confusion_matrix)
cm_button.pack(pady=10)

exit_button = tb.Button(root, text="Exit", bootstyle=DANGER, command=root.quit)
exit_button.pack(pady=10)

root.mainloop()
