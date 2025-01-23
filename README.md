# Machine Learning Mini-Projects with Tkinter GUI

## 📌 Overview
This repository contains six mini-projects that implement various machine learning models using **Python, Tkinter (GUI), and Scikit-Learn**. Each project demonstrates a different ML technique, such as **regression, classification, clustering, and visualization**.

---

## 📂 Project List

### **1️⃣ Real Estate Price Prediction (Linear Regression)**
- **Dataset:** `homeprices.csv` (Columns: `Area`, `Price`)
- **Model:** Linear Regression
- **Functionality:**
  - Predicts house price based on area.
  - Displays the regression equation.
  - Plots a regression line over data points.
  - Allows user input for prediction.
- **GUI Features:**
  - Input field for `Area`.
  - Button to predict the price.
  - Graph displaying regression results.

---

### **2️⃣ Startup Profit Prediction (Multiple Linear Regression)**
- **Dataset:** `50_Startups.csv` (Features: `R&D Spend`, `Marketing Spend`, `State`, `Profit`)
- **Model:** Multiple Linear Regression
- **Functionality:**
  - Predicts the profit of a startup based on its expenses.
  - Encodes categorical variables (`State`).
  - Standardizes features before training.
- **GUI Features:**
  - Button to train the model.
  - Button to display a scatter plot of actual vs. predicted profits.
  - Model accuracy (Mean Squared Error) displayed as an alert.

---

### **3️⃣ Salary Prediction Using Decision Trees**
- **Dataset:** `salaries.csv` (Columns: `company`, `job`, `degree`, `salary_more_than_100k`)
- **Model:** Decision Tree Classifier
- **Functionality:**
  - Predicts if salary is **more or less than $100K** based on `company`, `job`, and `degree`.
  - Uses Label Encoding for categorical features.
  - Evaluates model performance using accuracy and confusion matrix.
- **GUI Features:**
  - Dropdown menus for selecting `Company`, `Job`, and `Degree`.
  - Button to predict salary category.
  - Button to show model accuracy and confusion matrix.
  - Button to visualize the **Decision Tree**.

---

### **4️⃣ K-Nearest Neighbors (KNN) Classifier with GUI**
- **Dataset:** `user+data.csv`
- **Model:** K-Nearest Neighbors (KNN)
- **Functionality:**
  - Trains a KNN classifier to classify users based on features.
  - Standardizes data before training.
  - Displays a confusion matrix and decision boundary.
- **GUI Features:**
  - Button to display **Confusion Matrix**.
  - Button to plot the **Decision Boundary**.

---

### **5️⃣ KMeans Clustering App**
- **Dataset:** **Synthetic dataset** generated using `make_blobs()`.
- **Model:** KMeans Clustering
- **Functionality:**
  - Allows user to input the number of clusters.
  - Trains a KMeans clustering model.
  - Visualizes clusters in **2D and 3D**.
- **GUI Features:**
  - Input field for specifying **number of clusters**.
  - Button to **run KMeans**.
  - Button to **toggle between 2D and 3D** views.

---

### **6️⃣ Heatmap and Hierarchical Clustering Visualization**
- **Functionality:**
  - Uses heatmaps and hierarchical clustering to analyze patterns in matrices.
  - Uses dendrograms to visualize hierarchical relationships.
- **GUI Features:**
  - Heatmap visualization.
  - Dendrogram (Hierarchical Clustering Tree).
  - Color-coded clustering visualization.

---

## ⚙️ Installation & Usage
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/MehdiDinari/Some-Machine-Learning-algorithme-project.git
cd ml-tkinter-projects
```

### **2️⃣ Install Dependencies**
Make sure you have Python installed (>=3.7), then run:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run a Project**
Each project is standalone. Run any of the scripts using:
```bash
python project_name.py
```
For example:
```bash
python real_estate_price.py
```

---

## 🔧 Technologies Used
- **Python** (Core Programming Language)
- **Tkinter** (Graphical User Interface)
- **Scikit-Learn** (Machine Learning Models)
- **Matplotlib** (Data Visualization)
- **Pandas & NumPy** (Data Processing)

---

## 🚀 Future Enhancements
✅ Combine all mini-projects into a **single Tkinter application**.  
✅ Add **real-time data loading** from user-uploaded CSV files.  
✅ Improve UI/UX with **better layout and color schemes**.  
✅ Implement **model evaluation metrics** in each project.

---

## 📜 License
This project is **open-source** under the MIT License. You are free to use, modify, and distribute it.

---

## 🙌 Contributing
Feel free to fork this repository and submit a **pull request** if you want to improve it! 🚀

