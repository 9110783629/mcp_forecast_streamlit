# GRU-Based Forecasting App for Market Clearing Price (MCP)

This project is a forecasting web application built using a GRU (Gated Recurrent Unit) deep learning model.  
It predicts the Market Clearing Price (MCP) for the Indian Energy Exchange (IEX) Day-Ahead Market.

The application is deployed using **Flask** and can be hosted using platforms like **Render** or **AWS EC2**.

---

##  Project Overview

-  **Goal**: Real-time demand-supply balancing for power trading
-  **Target Variable**: Market Clearing Price (MCP)
-  **Best Model**: GRU (Gated Recurrent Unit)
-  **Forecast Interval**: 15-minute intervals
-  **Forecast Horizon**: January 2025 (1 month)

---

## 🗂️ Files Included

| File Name           | Description                                      |
|---------------------|--------------------------------------------------|
| `app.py`            | Flask web app to serve GRU predictions           |
| `best_model_gru.h5` | Trained GRU model file                           |
| `DAM__IND.csv`      | Input dataset used by the model                  |
| `requirements.txt`  | Python dependencies for the app                  |
| `render.yaml`       | Configuration file for Render deployment         |

---

##  Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gru-forecasting-app.git
   cd gru-forecasting-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

---

##  Deployment (Render)

1. Upload all files to a public GitHub repo
2. Create a new **Web Service** on [Render](https://render.com/)
3. Connect your repo and use:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`

---

##  Forecast Output

The app displays the 15-minute interval MCP forecasts for January 2025 using the trained GRU model.

---

## ‍ Author

- Name: *T.Chinni Krishna*
- Project: *Real-Time Demand-Supply Balancing for Power Trading*
