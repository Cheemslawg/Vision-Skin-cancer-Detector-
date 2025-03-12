import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load both models
classification_model = load_model('models/classification_model.h5')
abnormality_model = load_model('models/abnormality_detection.h5')

def process_image(file_path):
    # Abnormality Detection (Segmentation)
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (256, 256))
    mask = abnormality_model.predict(np.expand_dims(img_resized/255., axis=0))[0]
    
    # Classification
    classification_img = cv2.resize(img, (224, 224))
    classification_img = np.expand_dims(classification_img, axis=0)/255.
    prediction = classification_model.predict(classification_img)
    
    return mask, prediction

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Get both results
            mask, prediction = process_image(file_path)
            
            # Save visualization
            result_path = os.path.join('static', 'results', filename)
            visualize_abnormalities(file_path, mask, result_path)
            
            return render_template('result.html',
                                 result=class_names[np.argmax(prediction)],
                                 confidence=np.max(prediction),
                                 image_path=result_path)
            
        except Exception as e:
            return render_template('error.html', error=str(e))