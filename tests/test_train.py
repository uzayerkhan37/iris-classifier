import os
import pickle

def test_model_exists():
    assert os.path.exists("outputs/iris_model.pkl")

def test_model_predictions():
    with open("outputs/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    dummy_data = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(dummy_data)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]
