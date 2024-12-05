from src.feature_extraction import load_base_model
from joblib import load
import os

MODEL_DIR = "models"

class ModelLoader:
    _models = None

    @classmethod
    def load_models(cls):
        if cls._models is None:
            print("Loading models...")

            # Load the models only once
            resnet_model = load_base_model("resnet")
            vggnet_model = load_base_model("vggnet")
            xception_model = load_base_model("xception")
            ensemble_model = load(os.path.join(MODEL_DIR, "ensemble_model.joblib"))

            cls._models = (resnet_model, vggnet_model, xception_model, ensemble_model)
            print("Models loaded successfully!")
        return cls._models
