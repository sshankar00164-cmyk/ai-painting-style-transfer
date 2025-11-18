from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64

app = Flask(__name__)
CORS(app)

# Load pre-trained style transfer model
model = None
try:
    import tensorflow_hub as hub
    print("Loading TensorFlow Hub model...")
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Could not load model: {e}")

def load_img(img_bytes, max_dim=512):
    """Load and preprocess image"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        
        # Get original size
        width, height = img.size
        
        # Calculate new dimensions
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    try:
        tensor = np.array(tensor)
        if len(tensor.shape) > 3:
            tensor = tensor[0]
        tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(tensor)
    except Exception as e:
        raise ValueError(f"Error converting tensor: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "AI Painting Style Transfer API",
        "version": "1.0",
        "endpoints": {
            "health": "/health (GET)",
            "style_transfer": "/style-transfer (POST)"
        },
        "model_loaded": model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "model_loaded": model is not None
    })

@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    """Apply style transfer to content image"""
    try:
        # Validate request
        if 'content' not in request.files:
            return jsonify({"error": "Content image is required"}), 400
        if 'style' not in request.files:
            return jsonify({"error": "Style image is required"}), 400
        
        content_file = request.files['content']
        style_file = request.files['style']
        
        if content_file.filename == '':
            return jsonify({"error": "Content image filename is empty"}), 400
        if style_file.filename == '':
            return jsonify({"error": "Style image filename is empty"}), 400
        
        # Read image bytes
        content_bytes = content_file.read()
        style_bytes = style_file.read()
        
        if len(content_bytes) == 0:
            return jsonify({"error": "Content image is empty"}), 400
        if len(style_bytes) == 0:
            return jsonify({"error": "Style image is empty"}), 400
        
        print(f"Processing: {content_file.filename} + {style_file.filename}")
        
        # Load and preprocess images
        content_image = load_img(content_bytes)
        style_image = load_img(style_bytes)
        
        # Apply style transfer
        if model is not None:
            print("Applying neural style transfer...")
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            result_image = tensor_to_image(stylized_image.numpy())
        else:
            print("Model not available, using simple blend...")
            # Fallback method
            content_pil = Image.open(io.BytesIO(content_bytes)).convert('RGB')
            style_pil = Image.open(io.BytesIO(style_bytes)).convert('RGB')
            style_pil = style_pil.resize(content_pil.size, Image.LANCZOS)
            result_image = Image.blend(content_pil, style_pil, alpha=0.5)
        
        # Convert to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_str}"
        })
        
    except ValueError as ve:
        print(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI Painting Style Transfer Server")
    print("="*60)
    print(f"Status: Running")
    print(f"URL: http://localhost:5000")
    print(f"Model: {'Loaded' if model else 'Not Loaded (using fallback)'}")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)