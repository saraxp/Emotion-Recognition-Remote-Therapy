import os
import logging
from huggingface_hub import hf_hub_download

# Set up basic logging to see output in Render's build logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Read from Environment Variables (The standard way on Render)
REPO_ID = os.environ.get("HF_REPO_ID")
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME")
HF_TOKEN = os.environ.get("HF_TOKEN") # Optional: for private models

# --- Validation ---
if not REPO_ID or not MODEL_FILENAME:
    logger.error("Error: HF_REPO_ID and HF_MODEL_FILENAME environment variables must be set.")
    exit(1) # Exit with an error to fail the build

logger.info(f"Starting model download for repo: {REPO_ID}, file: {MODEL_FILENAME}")

# --- Download ---
try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=HF_TOKEN # Will be None if not set, which is fine for public repos
    )
    
    logger.info(f"Model successfully downloaded. Path: {model_path}")
    logger.info("This model is now cached and will be available to your Start Command.")

except Exception as e:
    logger.error(f"FATAL: Failed to download model: {e}")
    exit(1) # Fail the build

logger.info("Model download script finished successfully.")