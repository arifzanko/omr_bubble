from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from pipelines.training_pipelines import train_pipeline
from ultralytics import YOLO
import os
import base64
from PIL import Image

current_folder = os.getcwd()
model = YOLO(current_folder + '/shade_v6.pt')

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

APP_HOST = "0.0.0.0"
APP_PORT = 8080

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    train_pipeline()
    return "Training Successfull!!" 

@app.route("/")
def home():
    return render_template("index.html")


def check_lorek(image):
    try:
        flag = 0
        highest_probability = 0.0
        class_id = ''
        results = model.predict(source=image, save=True)
        result = results[0]
        
        if len(result) == 0:
            return ""
            
        for i, box in enumerate(result.boxes):
            class_id = result.names[box.cls[0].item()]
            probability = box.conf.item()
            print('class_id: ', class_id, ' probability:', probability)
            if probability > highest_probability:
                highest_probability = probability
                final_class_id = class_id

        print('Final class_id: ', final_class_id, ' Probability: ', highest_probability)
        if final_class_id == 'shade':
            flag = 1
        return flag

    except Exception as e:
        if "OMRB_NOT_FOUND:OMR bubble not found" in str(e) or "DPLICATE_OMR:Duplicate OMR found" in str(e):
            raise e  # Re-raise the specific exception
        else:
            raise Exception("OMR_ERR: OMR Error")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        image_predict = Image.open(current_folder+'/data'+'/inputImage.jpg')
        check_lorek(image_predict)
        #os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64("runs/detect/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf runs/detect/predict")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT, debug=True)