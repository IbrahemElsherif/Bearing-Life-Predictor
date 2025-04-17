Here's a clean and professional `README.md` for your **Bearing RUL Prediction** project that reflects your full workflow—local modular development, MLflow tracking, and Azure ML integration:

---

```markdown
# Bearing RUL Prediction

A machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of ball bearings using sensor data (vibration, temperature, speed, and force). The project was developed modularly with local MLflow tracking and later integrated into Azure ML for scalable training and inference.

---

## 🔧 Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── features/       # Feature engineering (e.g., rolling stats, FFT)
├── models/         # Model training and evaluation
├── prediction/     # Inference utilities
├── utils/          # Config loading, logging, and MLflow helpers
notebooks/          # Jupyter notebooks for experimentation
config.yaml         # Central configuration
```

---

## 🧪 Workflow Overview

1. **Modular Codebase**: Clean separation of data, features, models, and prediction logic.
2. **Local MLflow Tracking**: Run experiments locally and log parameters, metrics, and artifacts.
3. **Notebook-to-Script Conversion**: Prototyped in notebooks, converted to CLI scripts (`combine_data.py`, `train_model.py`, etc.).
4. **Azure ML Integration**:
   - Data stored in Azure Blob Storage
   - ML pipelines run modular stages: data prep → feature engineering → training → RUL prediction
   - Models tracked in Azure-backed MLflow and registered for batch/real-time inference

---

## 🚀 How to Run Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run MLflow tracking server (optional)**:
   ```bash
   mlflow ui
   ```

3. **Run data combination**:
   ```bash
   python src/data/combine_sensor_data.py
   ```

4. **Train model**:
   ```bash
   python src/models/train_model.py --config config.yaml
   ```

---

## 📊 Dataset

This project uses the [PRONOSTIA bearing dataset](https://www.femto-st.fr/en/Research-departments/AS2M/Software/PRONOSTIA) from the IEEE PHM 2012 Prognostic Challenge, containing sensor readings (vibration, temperature, speed, and force) collected under accelerated degradation.

---

## ✅ Features

- Rolling statistics on vibration signals
- Temperature trends and rate of change
- Operating condition inputs (speed, force, torque)
- Automatic feature selection based on importance
- XGBoost / Random Forest for RUL regression

---

## 📈 Sample Results

| Model           | RMSE   | MAE   | R²     |
|----------------|--------|-------|--------|
| XGBoost        | 8.62   | 5.61  | 0.9948 |
| Random Forest  | 10.41  | 5.49  | 0.9925 |

---

## ☁️ Azure ML Deployment

- Modular scripts submitted as Azure ML pipeline steps
- Data ingested from Blob Storage
- Final models registered and served from Azure ML Workspace
- Supports batch inference on new bearing data

---

## 📂 Output Files

- `data/processed/all_acc_data.csv`
- `data/processed/all_temp_data.csv`
- Trained models stored in `mlruns/` or Azure ML registry
- Evaluation metrics logged per run

---

## 🧠 Future Improvements

- Add frequency-domain features (Wavelet, FFT)
- Time-aware models (e.g., LSTM)
- Real-time streaming inference setup (IoT edge or Azure Function)

---

## 🧑‍💻 Contributors

- Team: [Your Names Here]

---

## 📜 License

MIT License
```

---

Would you like me to save this as a file or include project badges (e.g., Python version, MLflow)?