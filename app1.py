from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2     



app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_tooth_density(tooth):
    ret,thresh1 = cv2.threshold(tooth,160,200,cv2.THRESH_BINARY)
    temp = cv2.dilate(thresh1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    titles = ['Original Image','BINARY']
    images = [tooth, thresh1]
    print(len(contours))
    cv2.drawContours(tooth, contours, -1, (0,255,0), 3)
    largestCnt = []
    for cnt in contours:
        if (len(cnt) > len(largestCnt)):
            largestCnt = cnt
    # Determine center of area of largest contour
    M = cv2.moments(largestCnt)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Initiale mask for flood filling
    width, height = temp.shape
    mask = img2 = np.ones((width + 2, height + 2), np.uint8) * 255
    mask[1:width, 1:height] = 0
    # Generate intermediate image, draw largest contour, flood filled
    temp = np.zeros(temp.shape, np.uint8)
    temp = cv2.drawContours(temp, largestCnt, -1, 255, cv2.FILLED)
    cv2.imwrite('static/2.jpg',temp)
    _, temp, mask, _ = cv2.floodFill(temp, mask, (x, y), 255)
    for i in range(2):
        cv2.imwrite(f'static/{i}.jpg',images[i])
    return len(contours)
        

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/classify",methods=['POST'])
def classify():
    
    image = request.files['file']
    image.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(image.filename)))
    print(UPLOAD_FOLDER+image.filename)
    img=cv2.imread(UPLOAD_FOLDER+'/'+image.filename,0)
    density = get_tooth_density(img)
    return render_template('classification.html',density=density)


if __name__=='__main__':
    app.run(debug=True)
