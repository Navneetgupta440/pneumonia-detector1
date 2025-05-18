
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Prediction function
def predict_pneumonia(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    prediction = model.predict(img_array)
    return "🫁 Pneumonia Detected" if prediction[0][0] > 0.5 else "✅ Normal Lungs"

# Gradio UI with enhanced styling
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #005792;">🧠 Pneumonia Detection from Chest X-rays</h1>
        <p style="text-align: center;">Upload a chest X-ray image to detect signs of pneumonia using a deep learning model.</p>
        <hr>
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Chest X-ray")
            submit_button = gr.Button("🔍 Analyze")
        with gr.Column():
            output_label = gr.Label(label="Result")

    submit_button.click(fn=predict_pneumonia, inputs=image_input, outputs=output_label)

    gr.Markdown(
        """
        <hr>
        <p style="text-align: center; font-size: 14px;">Made with ❤️ by Sachin | Powered by TensorFlow & Gradio</p>
        """
    )

demo.launch()
