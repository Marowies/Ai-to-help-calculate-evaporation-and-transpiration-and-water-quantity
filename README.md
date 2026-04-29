# 🌌 Cosmic Irrigation System

![Cosmic UI](https://img.shields.io/badge/Design-Cosmic_System-blueviolet?style=for-the-badge)
![AI Powered](https://img.shields.io/badge/AI-XGBoost_Powered-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A futuristic, AI-powered irrigation prediction platform designed to optimize water usage in agriculture. Featuring a high-contrast, sci-fi inspired dashboard and real-time environmental data processing.

---

## 🚀 Features

- **Precision Prediction:** Uses XGBoost models to calculate ETc (Crop Evapotranspiration) and specific water requirements.
- **Cosmic UI/UX:** A unique futuristic aesthetic with vibrant neon accents and immersive spatial elements.
- **Dynamic Dashboard:** Visualize model performance, feature importance, and historical data comparisons.
- **Intelligent Crop Mapping:** Automatically updates crop coefficients (Kc) based on growth stages (Seedling, Vegetative, Flowering, Maturity).
- **Mobile Responsive:** Optimized for both desktop and mobile field use.

---

## 🛠️ Tech Stack

### Frontend
- **React 18** (Vite)
- **Recharts** (Data Visualization)
- **Vanilla CSS** (Cosmic Design Tokens)
- **Audiowide & JetBrains Mono Fonts**

### Backend
- **Python 3.11**
- **Flask** (API Framework)
- **Scikit-Learn & XGBoost** (ML Models)
- **Joblib** (Model Serialization)

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Marowies/Ai-to-help-calculate-evaporation-and-transpiration-and-water-quantity.git
cd Ai-to-help-calculate-evaporation-and-transpiration-and-water-quantity
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
python api_v2.py
```

### 3. Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

---

## 🌐 Deployment

The project is ready for deployment on **Render** or **Railway**.

1. **Backend:** Deploy `api_v2.py` as a Web Service.
2. **Frontend:** Deploy as a Static Site, ensuring `VITE_API_URL` environment variable points to your backend URL.

---

## 👨‍💻 Developed By

**Marwan _ewis**

Project created to merge advanced Artificial Intelligence with sustainable agricultural practices.

---

© 2026 COSMIC IRRIGATION SYSTEM | ALL RIGHTS RESERVED
