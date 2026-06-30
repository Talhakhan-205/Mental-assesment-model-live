# 🧠 MindVault — Mental Health Crisis Predictor

A Streamlit web app that predicts mental health treatment need using ML.

## Quick Start (VS Code)

1. Open this folder in VS Code
2. Open Terminal → `Ctrl + `` ` ``
3. Run:
```bash
pip install -r requirements.txt
python train_improved_model.py
streamlit run app.py
```

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app |
| `train_improved_model.py` | Train/retrain the ML model |
| `mental_health_model.pkl` | Pre-trained model |
| `requirements.txt` | Python dependencies |
| `PIPELINE_GUIDE.md` | Full technical guide |

## Note
Put `Mental Health Dataset.csv` in this folder before retraining.
