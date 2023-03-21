from flask import Flask,request,jsonify,abort,make_response,render_template,json
from json import dumps
import os
from keras_preprocessing import image
import joblib
import dlib
# from firebase import firebase
import cv2
import numpy as np
# from skimage.transform import resize
from PIL import Image
from PIL import Image as im
from vggface import extract, load_model,load_coefficients
from sklearn.svm import SVC
app = Flask(__name__)


@app.route('/api/upload', methods=['POST'])
def handle_form():
    files = request.files
    file = files.get('file')
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # cv2.imwrite('C:/Users/User/Desktop/ml_final_web/sandbox/img3'+ '.png', img)

    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    for num, face in enumerate(faces):
        # 计算矩形框大小
        height = face.bottom() - face.top()
        width = face.right() - face.left()

        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[face.top() + i][face.left() + j]
        # cv2.imwrite('C:/Users/User/Desktop/ml_final_web/sandbox/img5'+ '.png', img_blank)
        
        # min_params = img_blank.min()
        # max_params = img_blank.max()
        # img_blank = 255*(img_blank - min_params) / (max_params- min_params) #normalized to (0,1)
        # resized = resize(img_blank, (224, 224))
        resized = cv2.resize(img_blank, (224, 224), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('C:/Users/User/Desktop/ml_final_web/sandbox/resize1'+ '.png', resized)
        out = resized.round()
        data = im.fromarray(np.uint8(out))
        data = data.convert('RGB')
        # data.save('C:/Users/User/Desktop/ml_final_web/sandbox/img1'+ '.png','PNG')
        # img_blank.save('C:/Users/User/Desktop/ml_final_web/sandbox/img2'+ '.png','PNG')
        # cv2.imwrite('C:/Users/User/Desktop/ml_final_web/sandbox/img2'+ '.png', img_blank)
        

    
    # feat = extract(data)
    feat = extract(resized)
    feat = feat.reshape(1,-1)
    # np.array(feat).shape()
    # clf = SVC(probability=True)
    # load_coefficients(clf, "model_coefficient.h5")
    clf = load_model()
    y_pred = (clf.predict_proba(feat)[0][1] >= 0.6477754765348217).astype(int)
    # y_pred = clf.predict_proba(feat)
    # y_pred = (clf.predict_proba(feat)[:,1] >= 0.6477754765348217).astype(int)
    feat = feat.tolist()
    y_pred = y_pred.tolist()
    # # print(y_pred)
    # return make_response(dumps(y_pred))
    return jsonify({'val':y_pred})


if __name__ == '__main__':
    app.run(debug=True)



