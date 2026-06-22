# Iris Flower Classifier

A professional, end-to-end Machine Learning project that classifies Iris flower species using a Random Forest model. This repository demonstrates clean project structuring, virtual environments, interactive data analysis, and reproducible training scripts.

## 📊 Project Structure

- `data/`: Contains dataset references (data is loaded dynamically via `scikit-learn`).
- `notebooks/`: Interactive Jupyter Notebook (`iris_model.ipynb`) used for exploration and prototyping.
- `src/`: Core Python modules, including `train.py` for training the model via the command line.
- `tests/`: Basic testing suite using `pytest`.
- `outputs/`: Automatically stores generated evaluation charts and trained model artifacts.

## 🚀 Getting Started

### 1. Prerequisites & Setup

Clone the repository and ensure you are in the project root directory. Then, set up your virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```
