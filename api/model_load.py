import joblib

def load_model(model_path):
    """Load the trained model from the specified path."""
    try:
        # Load the model using joblib
        model = joblib.load(model_path)
        return model
    except EOFError:
        print("Error: The model file is empty or corrupted.")
        return None
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return None
