from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import pytesseract
from PIL import Image

app = FastAPI()

# Load the TensorFlow model
model = tf.keras.models.load_model('components/model')

# Define the OCR function
def perform_ocr(image):
    image = image.convert('L')  # Convert the image to grayscale
    text = pytesseract.image_to_string(image)  # Perform OCR
    return text

# Define the prediction route
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save the uploaded image
    with open('temp.jpg', 'wb') as f:
        f.write(await file.read())

    # Open the image
    image = Image.open('temp.jpg')

    # Perform OCR
    text = perform_ocr(image)

    # Preprocess the image for prediction
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Add batch dimension

    # Make the prediction using the TensorFlow model
    prediction = model.predict(image_array)
    predicted_class = tf.argmax(prediction, axis=1)[0]

    # Return the prediction and OCR result
    return {"prediction": int(predicted_class), "ocr_text": text}