from sklearn.base import BaseEstimator
import numpy as np
import sklearn

def score(text: str, model, threshold: float) -> tuple[bool, float]:
    print(f"Scoring text: {text}")
    probabilities = model.predict_proba([text])
    propensity = probabilities[0, 1]
    prediction = propensity >= threshold
    print(f"Propensity: {propensity}, Prediction: {prediction}")
    return bool(prediction), propensity
