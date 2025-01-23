import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics

col_names = ['company', 'job', 'degree', 'salary_more_than_100k']
data = pd.read_csv('salaries.csv', names=col_names)

label_encoder_company = preprocessing.LabelEncoder()
label_encoder_job = preprocessing.LabelEncoder()
label_encoder_degree = preprocessing.LabelEncoder()

data['company'] = label_encoder_company.fit_transform(data['company'])
data['job'] = label_encoder_job.fit_transform(data['job'])
data['degree'] = label_encoder_degree.fit_transform(data['degree'])

company_labels = list(label_encoder_company.classes_)
job_labels = list(label_encoder_job.classes_)
degree_labels = list(label_encoder_degree.classes_)

feature_cols = ['company', 'job', 'degree']
x = data[feature_cols]
y = data['salary_more_than_100k']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

root = tk.Tk()
root.title("Prédiction de Salaire avec un Arbre de Décision")
root.geometry("600x500")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Modèle de Prédiction de Salaire", font=("Arial", 16, "bold"), bg="#f0f0f0",
                       fg="#333")
title_label.pack(pady=10)


def show_accuracy():
    messagebox.showinfo("Précision du modèle", f"Précision: {accuracy:.2f}")


def show_confusion_matrix():
    messagebox.showinfo("Matrice de Confusion", f"{conf_matrix}")


def plot_decision_tree():
    plt.figure(figsize=(12, 6))
    plot_tree(clf, feature_names=feature_cols, class_names=[str(cls) for cls in np.unique(y_train)], filled=True)
    plt.title("Arbre de Décision - Salaire")
    plt.show(block=False)  # 🔧 Empêche la fenêtre Tkinter de se figer


def predict_salary():
    try:
        company_index = label_encoder_company.transform([company_var.get()])[0]
        job_index = label_encoder_job.transform([job_var.get()])[0]
        degree_index = label_encoder_degree.transform([degree_var.get()])[0]

        input_data = pd.DataFrame([[company_index, job_index, degree_index]], columns=feature_cols)

        prediction = clf.predict(input_data)[0]

        result_text.set(f"Prédiction : {'>100K' if prediction == 1 else '≤100K'}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")


accuracy_btn = tk.Button(root, text="Afficher la Précision", command=show_accuracy, font=("Arial", 12), bg="#4CAF50",
                         fg="white")
accuracy_btn.pack(pady=5)

conf_matrix_btn = tk.Button(root, text="Afficher Matrice de Confusion", command=show_confusion_matrix,
                            font=("Arial", 12), bg="#f44336", fg="white")
conf_matrix_btn.pack(pady=5)

tree_btn = tk.Button(root, text="Afficher l'Arbre de Décision", command=plot_decision_tree, font=("Arial", 12),
                     bg="#008CBA", fg="white")
tree_btn.pack(pady=5)

predict_label = tk.Label(root, text="Entrez les informations :", font=("Arial", 12, "bold"), bg="#f0f0f0")
predict_label.pack(pady=10)

company_var = tk.StringVar(value=company_labels[0])
job_var = tk.StringVar(value=job_labels[0])
degree_var = tk.StringVar(value=degree_labels[0])
result_text = tk.StringVar()

company_label = tk.Label(root, text="Entreprise :", font=("Arial", 12), bg="#f0f0f0")
company_label.pack()
company_dropdown = ttk.Combobox(root, textvariable=company_var, values=company_labels, font=("Arial", 12),
                                state="readonly")
company_dropdown.pack()

job_label = tk.Label(root, text="Poste :", font=("Arial", 12), bg="#f0f0f0")
job_label.pack()
job_dropdown = ttk.Combobox(root, textvariable=job_var, values=job_labels, font=("Arial", 12), state="readonly")
job_dropdown.pack()

degree_label = tk.Label(root, text="Diplôme :", font=("Arial", 12), bg="#f0f0f0")
degree_label.pack()
degree_dropdown = ttk.Combobox(root, textvariable=degree_var, values=degree_labels, font=("Arial", 12),
                               state="readonly")
degree_dropdown.pack()

predict_btn = tk.Button(root, text="Prédire le Salaire", command=predict_salary, font=("Arial", 12), bg="#ff9800",
                        fg="white")
predict_btn.pack(pady=10)

result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333")
result_label.pack(pady=10)

root.mainloop()
