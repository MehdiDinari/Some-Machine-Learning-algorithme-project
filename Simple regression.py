import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv('homeprices.csv')

df.columns = df.columns.str.strip()

expected_columns = ['Area', 'Price']
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"⚠️ Erreur : Les colonnes attendues {expected_columns} sont introuvables.")

reg = LinearRegression()
X = df[['Area']]
y = df['Price']
reg.fit(X, y)

X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = reg.predict(X_pred)

root = tk.Tk()
root.title("Application de Régression Linéaire")
root.geometry("800x600")

equation_label = tk.Label(root, text=f"📈 Équation : Prix = {reg.coef_[0]:.2f} * Surface + {reg.intercept_:.2f}", font=("Arial", 12))
equation_label.pack(pady=10)

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df['Area'], df['Price'], color='red', marker='*', label="Données réelles")
ax.plot(X_pred, y_pred, color='blue', label="Ligne de régression")
ax.set_xlabel('Surface (sq ft)')
ax.set_ylabel('Prix (DH)')
ax.set_title('Régression Linéaire: Surface vs Prix')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

entry_label = tk.Label(root, text="Entrez la surface (sq ft) :", font=("Arial", 12))
entry_label.pack(pady=5)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

def predict_price():
    try:
        area = float(entry.get())
        predicted_price = reg.predict([[area]])[0]
        result_label.config(text=f"🔮 Prix estimé : {predicted_price:,.2f} DH", fg="blue")
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer une valeur numérique valide.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

predict_button = tk.Button(root, text="Prédire le Prix", font=("Arial", 12), command=predict_price)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

quit_button = tk.Button(root, text="Quitter", font=("Arial", 12), command=root.quit)
quit_button.pack(pady=10)

root.mainloop()
