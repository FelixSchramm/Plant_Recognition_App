"""Backend helpers for the interactive plant disease diagnosis page.

This module bundles everything the ``7_Live_Diagnosis`` page needs to turn an
uploaded leaf image into a prediction: loading the class names, loading the
trained MobileNetV2 model from the Hugging Face Hub, pre-processing the image
and running the prediction.

The advanced model already contains the MobileNetV2 pre-processing as a
``Lambda(preprocess_input)`` layer, so images are fed as raw ``float32`` pixel
values in the 0-255 range (no external ``preprocess_input``).

If no model is configured (or the download fails), a deterministic mock
predictor is used instead. This keeps the page fully usable as a demo while the
real model weights are being hosted on the Hugging Face Hub.
"""

import glob
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image

# --- Paths -----------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASS_NAMES_PATH = os.path.join(_SCRIPT_DIR, "class_names.json")
_EXAMPLES_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "images_grad_cam", "advanced_model")
)

# --- Model configuration ---------------------------------------------------
# Input size expected by the advanced MobileNetV2 model.
IMAGE_SIZE = (224, 224)

# Hugging Face Hub defaults. Override them in ``.streamlit/secrets.toml``::
#
#     hf_model_repo = "felixschramm/plant-disease-mobilenetv2"
#     hf_model_filename = "..._fine_tuned_int_phase_model.keras"
#     hf_token = "hf_..."   # only required for a private repo
DEFAULT_HF_REPO = "felixschramm/plant-disease-mobilenetv2"
DEFAULT_HF_FILENAME = "2025_07_31_plant_classifier_fine_tuned_int_phase_model.keras"


@dataclass
class Prediction:
    """A single class prediction with its parsed, human readable parts.

    :param label: Raw class label, e.g. ``"Tomato___Late_blight"``.
    :param plant: Human readable plant name, e.g. ``"Tomato"``.
    :param disease: Human readable disease name or ``"healthy"``.
    :param is_healthy: ``True`` when the class denotes a healthy leaf.
    :param confidence: Predicted probability in the ``0..1`` range.
    """

    label: str
    plant: str
    disease: str
    is_healthy: bool
    confidence: float


def _get_secret(key, default=None):
    """Read a configuration value from ``st.secrets`` or the environment.

    Accessing ``st.secrets`` raises when no secrets file exists, so the lookup
    is guarded. As a fallback the upper-cased ``key`` is read from the process
    environment.

    :param key: Configuration key to look up.
    :param default: Value returned when the key is not configured.
    :return: The configured value or ``default``.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key.upper(), default)


@st.cache_data
def load_class_names():
    """Load the 38 class names in the exact training order.

    :return: List of raw class labels (``"Plant___Disease"``).
    """
    with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_label(label):
    """Split a raw class label into plant, disease and health status.

    ``"Tomato___Late_blight"`` becomes ``("Tomato", "Late blight", False)`` and
    ``"Apple___healthy"`` becomes ``("Apple", "healthy", True)``.

    :param label: Raw class label.
    :return: Tuple ``(plant, disease, is_healthy)``.
    """
    if "___" in label:
        plant_raw, disease_raw = label.split("___", 1)
    else:
        plant_raw, disease_raw = label, ""

    plant = plant_raw.replace("_", " ").strip()
    is_healthy = disease_raw.lower() == "healthy"
    disease = "healthy" if is_healthy else disease_raw.replace("_", " ").strip()
    return plant, disease, is_healthy


@st.cache_resource(show_spinner="Loading AI model...")
def load_diagnosis_model():
    """Load the diagnosis model, falling back to a mock predictor.

    Tries to download the trained Keras model from the Hugging Face Hub and
    load it. On any failure (no configuration, offline, download error) a mock
    backend is returned so the page keeps working as a demo.

    :return: Dict with ``kind`` (``"keras"`` or ``"mock"``), the loaded
        ``model`` when available and an ``error`` message when mocking.
    """
    repo = _get_secret("hf_model_repo", DEFAULT_HF_REPO)
    filename = _get_secret("hf_model_filename", DEFAULT_HF_FILENAME)
    token = _get_secret("hf_token", None)

    try:
        from huggingface_hub import hf_hub_download
        from tensorflow.keras.models import load_model

        model_path = hf_hub_download(repo_id=repo, filename=filename, token=token)
        # ``safe_mode=False`` allows loading the Lambda(preprocess_input) layer.
        model = load_model(model_path, safe_mode=False, compile=False)
        return {"kind": "keras", "model": model, "error": None}
    except Exception as exc:  # noqa: BLE001 - any failure falls back to mock
        return {"kind": "mock", "model": None, "error": str(exc)}


def is_mock_backend(model_bundle):
    """Return ``True`` when the given bundle is the mock predictor.

    :param model_bundle: Bundle returned by :func:`load_diagnosis_model`.
    :return: ``True`` if no real model is attached.
    """
    return model_bundle.get("kind") != "keras"


def preprocess_image(image, size=IMAGE_SIZE):
    """Resize an image to the model input and return a single-image batch.

    The MobileNetV2 pre-processing lives inside the model, so pixels are kept
    as raw ``float32`` values in the 0-255 range here.

    :param image: A ``PIL.Image`` instance.
    :param size: Target ``(width, height)`` for the model input.
    :return: ``numpy`` array of shape ``(1, height, width, 3)``.
    """
    resized = image.convert("RGB").resize(size)
    array = np.asarray(resized, dtype="float32")
    return np.expand_dims(array, axis=0)


def _mock_probabilities(image, num_classes):
    """Create deterministic pseudo-probabilities from an image.

    The same image always yields the same demo prediction because the random
    seed is derived from the image content. One class is boosted so there is a
    clear, confident-looking winner.

    :param image: A ``PIL.Image`` instance.
    :param num_classes: Number of output classes.
    :return: ``numpy`` probability vector that sums to 1.
    """
    thumbnail = image.convert("RGB").resize((32, 32))
    digest = hashlib.sha256(thumbnail.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)

    logits = rng.normal(size=num_classes)
    logits[rng.integers(num_classes)] += 4.0
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict(model_bundle, image, class_names, top_k=3):
    """Run the diagnosis and return the top predictions.

    :param model_bundle: Bundle returned by :func:`load_diagnosis_model`.
    :param image: A ``PIL.Image`` instance to classify.
    :param class_names: List of raw class labels in training order.
    :param top_k: Number of top predictions to return.
    :return: Tuple ``(predictions, probabilities)`` where ``predictions`` is a
        list of :class:`Prediction` sorted by descending confidence.
    """
    if is_mock_backend(model_bundle):
        probabilities = _mock_probabilities(image, len(class_names))
    else:
        batch = preprocess_image(image)
        probabilities = model_bundle["model"].predict(batch, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    predictions = []
    for index in top_indices:
        label = class_names[int(index)]
        plant, disease, is_healthy = parse_label(label)
        predictions.append(
            Prediction(
                label=label,
                plant=plant,
                disease=disease,
                is_healthy=is_healthy,
                confidence=float(probabilities[index]),
            )
        )
    return predictions, probabilities


def format_class_name(label):
    """Format a raw label for display, e.g. ``"Tomato (Late blight)"``.

    :param label: Raw class label.
    :return: Human readable string.
    """
    plant, disease, is_healthy = parse_label(label)
    if is_healthy:
        return f"{plant} (healthy)"
    return f"{plant} ({disease})"


@st.cache_data
def list_example_images(limit=6):
    """Collect a spread of ready-to-use example images with known labels.

    Uses the original (non Grad-CAM) images that ship with the repository so
    the page can be tried without uploading anything.

    :param limit: Maximum number of example images to return.
    :return: List of ``(display_name, file_path)`` tuples.
    """
    files = sorted(glob.glob(os.path.join(_EXAMPLES_DIR, "*_original_img*.png")))
    if not files:
        return []

    # Spread the selection across the available classes instead of taking the
    # first N files (which would all be apples).
    step = max(1, len(files) // limit)
    chosen = files[::step][:limit]

    examples = []
    for path in chosen:
        raw = os.path.basename(path).rsplit("_original_img", 1)[0]
        examples.append((format_class_name(raw), path))
    return examples
