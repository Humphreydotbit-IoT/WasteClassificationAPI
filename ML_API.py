from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load the model (replace with the directory containing your model)
MODEL_DIR = "/media/tun/X1/Research/FastAPI/my_model"
model = tf.saved_model.load(MODEL_DIR)

# Define an endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))  # Adjust size to match model's expected input
    image_array = np.array(image)

    # Preprocess the image here if required, e.g., scaling
    image_array = image_array / 255.0

    # Make a prediction
    predictions = model.predict(np.expand_dims(image_array, axis=0))

    # Return the result
    return {"predictions": predictions.tolist()}

# Note: To run the server, use the command:
# uvicorn your_script_filename:app --reload

