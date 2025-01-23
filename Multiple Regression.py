import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use('TkAgg')

class StartupPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Startup Profit Predictor")
        self.root.geometry("600x400")

        self.dataset = pd.read_csv("50_Startups.csv")
        self.X = None
        self.y = None
        self.model = None

        self.process_data()
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Train Model", command=self.train_model).pack(pady=10)
        tk.Button(self.root, text="Show Predictions", command=self.show_predictions).pack(pady=10)

    def process_data(self):
        X = self.dataset.iloc[:, :-1].values  # Independent variables
        y = self.dataset.iloc[:, -1].values  # Dependent variable

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.y = y

        messagebox.showinfo("Success", "Data processing completed!")

    def train_model(self):
        if self.X is None or self.y is None:
            messagebox.showwarning("Warning", "Please process the data first!")
            return

        X_train, X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        self.y_pred = self.model.predict(X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)
        messagebox.showinfo("Model Trained", f"Training completed!\nMSE: {mse:.2f}")

    def show_predictions(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(df.head())

        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue')
        plt.xlabel("Actual Profits")
        plt.ylabel("Predicted Profits")
        plt.title("Actual vs Predicted Profits")
        plt.show(block=True)  # Ensures proper visualization in Tkinter

if __name__ == "__main__":
    root = tk.Tk()
    app = StartupPredictorApp(root)
    root.mainloop()
