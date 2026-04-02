# 🏙️ Dalma Mall | Energy Intelligence Simulation Engine

### 🚀 The Mission: From Reactive to Predictive
In the extreme climate of Abu Dhabi, cooling accounts for the vast majority of operational expenditure. The **Dalma Energy Simulation Engine** transitions mall operations from a **Reactive** approach (responding to heat after it arrives) to a **Proactive** strategy (simulating demand 24-48 hours in advance). 

By leveraging atmospheric physics and human traffic patterns, this engine allows the engineering team to optimize the **Coefficient of Performance (COP)** and utilize the building's thermal mass as a strategic energy battery.

---

### 🧠 The Core Engine: Physics-Informed AI
Unlike standard "Black Box" models, this engine is **Lag-Independent**. It does not rely on "yesterday's data," which often fails during seasonal shifts. Instead, it utilizes **Atmospheric Enthalpy**—the true measure of thermal energy in the air.


---

### 💼 Business Value & ROI
1.  **Volumetric Energy Savings:** Identifying "High-COP" windows (typically 6:00 AM – 10:00 AM) to perform heavy cooling work when ambient temperatures are low, maximizing the cooling-per-Dirham ratio.
2.  **Peak Load Shaving:** Using the mall's concrete structure as a **Thermal Battery** to "charge" with cold energy during off-peak hours, flattening the 3:00 PM demand spike.
3.  **Asset Longevity:** Reducing mechanical "Load Shocks" on the chiller plant through predictive staging and soft-loading, extending the operational life of multi-million AED compressors.
4.  **Operational Certainty:** An early warning system for extreme weather events (45°C+ / High Humidity) that may exceed the plant's design capacity.

---

### 📊 Model Performance Metrics
The model utilizes a **Ridge Regression** pipeline with **Polynomial Features (Degree 2)** to accurately capture the non-linear relationship between humidity spikes and energy load.

| Metric | Power Demand (kWs) | Cooling Load (Tons) |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9584** | **0.9637** |
| **Mean Absolute Error** | 2377.42 | 2370.95 |

---

### 🛠️ Technical Stack
* **Core:** Python 3.10
* **Machine Learning:** Scikit-Learn (PolynomialFeatures, Ridge Regression, MultiOutputRegressor)
* **Data Processing:** Pandas, NumPy
* **Interface:** Streamlit (Interactive Dashboard)
* **Serialization:** Joblib

---

### 📥 Installation & Local Usage
1.  **Clone the Repository:**
    
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

### ⚠️ Disclaimer
*This project was developed during a technical internship at Dalma Mall. All operational data and results are proprietary. The simulation tool is intended for operational guidance and does not replace certified mechanical engineering inspections or BMS protocols.*
