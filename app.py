from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)

# Função para aplicar o filtro selecionado na imagem
def aplicar_filtro(image_data, nome_filtro):
    # Decodifica a imagem enviada no formato base64 (data URL)
    # image_data tem o formato "data:image/jpeg;base64,XXXXX..."
    encoded_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(encoded_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return None

    #aplica o filtro blur
    if nome_filtro == 'blur':
        img_com_filtro = cv2.GaussianBlur(image, (15, 15), 0)
    #aplica o filtro sharpen utilizando o kernel
    elif nome_filtro == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_com_filtro = cv2.filter2D(image, -1, kernel)
    #rotaciona a imagem de acordo com a matriz de rotaca
    elif nome_filtro == 'rotate':
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
        img_com_filtro = cv2.warpAffine(image, matrix, (w, h))
    else:
        return None

    #converte a imagem processada para jpg e retorna como BytesIO
    _, img_encoded = cv2.imencode('.jpg', img_com_filtro)
    return io.BytesIO(img_encoded.tobytes())

#rota principal para carregar o index
@app.route('/')
def index():
    return render_template('index.html')

#rota para processar a imagem e aplicar o filtro selecionado
@app.route('/process', methods=['POST'])
def process_image():
    image_data = request.form.get('image')
    nome_filtro = request.form.get('filter')

    processed_image = aplicar_filtro(image_data, nome_filtro)
    if processed_image:
        return send_file(processed_image, mimetype='image/jpeg')
    
    return "Erro ao processar imagem", 400

if __name__ == '__main__':
    app.run(debug=True)
