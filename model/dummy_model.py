import random

def predict_tooth_shade(image_path):
    """
    Dummy AI model that randomly selects a VITA shade.
    Replace this with real model inference later.
    """
    shades = ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'D2']
    return random.choice(shades)