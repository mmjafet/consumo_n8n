
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file
import numpy as np
import cv2
import requests
import os
import base64
from PIL import Image
from io import BytesIO
import uuid

app = Flask(__name__)

# Crear carpetas necesarias si no existen
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emotion_prediction')
def emotion_prediction():
    return render_template('emotion_prediction.html')

@app.route('/tumor_prediction')
def tumor_prediction():
    return render_template('tumor_prediction.html')

@app.route('/process_tumor_prediction', methods=['POST'])
def process_tumor_prediction():
    if 'file' not in request.files:
        return redirect(url_for('tumor_prediction'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('tumor_prediction'))
    
    # Guardar imagen original
    unique_id = str(uuid.uuid4())
    img_path = os.path.join('static/uploads', f'{unique_id}.jpg')
    file.save(img_path)
    
    # Procesar imagen
    image = Image.open(img_path).convert('RGB')
    processed_image = preprocess_image(image)
    
    # Llamar al webhook de n8n
    n8n_webhook_url = "https://susan2610.app.n8n.cloud/webhook/f2e95001-8627-4fba-9a28-ba5f98a9056a"
    instance_title = [f"Análisis de tumor - {unique_id}"]
    
    # Simular una predicción (en un caso real, esto vendría de un modelo ML)
    # Aquí hacemos un llamado a un servicio externo para obtener la predicción
    prediction = get_tumor_prediction(processed_image)
    
    payload = {
        "instance_title": instance_title,
        "prediction": float(prediction)
    }
    
    # Llamar al webhook de n8n
    n8n_response = requests.post(n8n_webhook_url, json=payload)
    
    tumor_detected = prediction > 0.5
    result_data = {
        'result': True,
        'prediction': prediction,
        'tumor_detected': tumor_detected,
        'original_img': img_path
    }
    
    # Si se detecta tumor, obtener segmentación
    if tumor_detected:
        # Enviar a la API de segmentación
        segmentation_url = "https://brain-models-v1.onrender.com/v1/models/ResUNet:predict"
        try:
            # Preparar imagen para segmentación (puede requerir preprocesamiento adicional)
            segmentation_image = preprocess_image(image, is_segmentation=True)
            
            # Llamar a la API de segmentación
            segmentation_response = requests.post(
                segmentation_url, 
                json={"instances": segmentation_image.tolist()}
            )
            
            if segmentation_response.status_code == 200:
                # Procesar respuesta de segmentación
                mask_data = np.array(segmentation_response.json()["predictions"][0])
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Guardar máscara
                mask_path = os.path.join('static/results', f'{unique_id}_mask.jpg')
                cv2.imwrite(mask_path, mask)
                
                # Crear superposición
                original_array = np.array(image)
                mask_resized = cv2.resize(mask, (original_array.shape[1], original_array.shape[0]))
                
                # Colorear la máscara
                mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
                
                # Superponer
                overlay = cv2.addWeighted(
                    cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR), 
                    0.7, 
                    mask_colored, 
                    0.3, 
                    0
                )
                
                # Guardar superposición
                overlay_path = os.path.join('static/results', f'{unique_id}_overlay.jpg')
                cv2.imwrite(overlay_path, overlay)
                
                # Añadir rutas a resultado
                result_data['mask_img'] = mask_path
                result_data['overlay_img'] = overlay_path
            
        except Exception as e:
            print(f"Error en segmentación: {str(e)}")
    
    return render_template('tumor_prediction.html', **result_data)

def preprocess_image(image, is_segmentation=False):
    """Preprocesa la imagen para los modelos."""
    if is_segmentation:
        resized = image.resize((256, 256))
    else:
        resized = image.resize((128, 128))
    
    img_array = np.array(resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_tumor_prediction(image_data):
    """
    Simula una predicción de tumor o conecta con un servicio real.
    En un caso real, esto conectaría con un modelo de ML.
    """
    # Esta es una simulación - en producción, conectarías con un modelo real
    # Aquí podrías usar un modelo local o llamar a un API
    try:
        # Ejemplo de llamada a un servicio de predicción
        classifier_url = "https://brain-models-v1.onrender.com/v1/models/tumor_classifier:predict"
        response = requests.post(
            classifier_url,
            json={"instances": image_data.tolist()}
        )
        
        if response.status_code == 200:
            prediction_value = response.json()["predictions"][0][0]
            return prediction_value
        else:
            # Si hay error, retornar un valor aleatorio para demostración
            return np.random.random()
    except:
        # En caso de error, generar un valor aleatorio para demostración
        return np.random.random()

if __name__ == '__main__':
    app.run(debug=True)