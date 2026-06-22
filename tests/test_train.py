import os
import subprocess

def test_training_script_generates_model():

    if os.path.exists("outputs/iris_model.joblib"):
        os.remove("outputs/iris_model.joblib")
        
    
    result = subprocess.run(["python", "src/train.py", "--estimators", "10"], capture_output=True, text=True)
    
    
    assert result.returncode == 0
    assert os.path.exists("outputs/iris_model.joblib") == True