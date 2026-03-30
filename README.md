# 🚗 Second-Hand Car Price Prediction

A Machine Learning-based web application that predicts the selling price of used cars based on various features like brand, model, fuel type, and vehicle specifications.

---

## 📌 Project Overview

This project uses a **Machine Learning pipeline** built with **Scikit-learn** and **XGBoost** to estimate car prices.
The model is deployed using **Streamlit**, providing an interactive user interface.

---

## 🚀 Features

* 🔍 Predict second-hand car prices instantly
* 🎯 Dynamic model selection based on brand
* ⚙️ Automated feature engineering
* 📊 Robust ML pipeline (preprocessing + model)
* 💻 Clean and user-friendly UI with Streamlit

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

## 📂 Project Structure

```
📁 Second-hand-price-prediction
│
├── app.py                      # Streamlit application
├── car_price_pipeline.pkl      # Trained ML pipeline
├── cardekho_dataset.csv        # Dataset
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd second-hand-price-prediction
```

### 2️⃣ Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

---

## 🧠 Model Details

* Algorithm: **XGBoost Regressor**
* Preprocessing:

  * OneHot Encoding for categorical features
  * Feature Engineering:

    * `km_per_year`
    * `power_to_engine`
* Target Transformation:

  * Log transformation using `log1p`

---

## 📊 Input Features

* Brand
* Model
* Vehicle Age
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission Type
* Mileage
* Engine
* Max Power
* Seats

---

## 📈 Output

* 💰 Predicted Selling Price (₹)

---

## ⚠️ Important Notes

* Ensure **same scikit-learn version** is used for training and deployment
* Keep `.pkl` files in the same directory as `app.py`
* Input data should match training format

---

## 🎯 Future Improvements

* 📊 Add price trend visualizations
* 🌐 Deploy on cloud (Streamlit Cloud / Render)
* 🧠 Improve model accuracy with more features
* 🖼️ Add car images based on model

---

## 🙌 Acknowledgements

* Dataset: CarDekho Dataset
* Libraries: Scikit-learn, XGBoost, Streamlit

---

## 👨‍💻 Author

Deep Timbaliya

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
