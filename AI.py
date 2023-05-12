from flask import Flask
import json
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from bson import json_util
import json
import cv2 
import numpy as np
from flask import Flask
import cv2 as cv
import numpy as np
from flask import Flask,jsonify
import json
from bson import json_util
import PopulateDb as popDb
import tkinter as tk
from utils.image_classifier import ImageClassifier, NO_FACE_LABEL
app = Flask(__name__)

@app.route("/objectdetection")
def live_object_count():

# @app.route("/facesummary",methods=['GET'])  #/<int:val>
# def recognition():
    best_match_index= int(0)
    
    video_capture = cv2.VideoCapture(0)
    
    Gaurav_image = face_recognition.load_image_file("./images/Gaurav.png")
    Gaurav_face_encoding = face_recognition.face_encodings(Gaurav_image)[0]

    Mahima_image = face_recognition.load_image_file("./images/Mahima.png")
    Mahima_face_encoding = face_recognition.face_encodings(Mahima_image)[0]

    Sardeep_image = face_recognition.load_image_file("./images/Sardeep.jpg")
    Sardeep_face_encoding = face_recognition.face_encodings(Sardeep_image)[0]

    Sandeep_image = face_recognition.load_image_file("./images/Sandeep.png")
    Sandeep_face_encoding = face_recognition.face_encodings(Sandeep_image)[0]

    Aakash_image = face_recognition.load_image_file("./images/Aakash.png")
    Aakash_face_encoding = face_recognition.face_encodings(Aakash_image)[0]

    Abhishek_image = face_recognition.load_image_file("./images/Abhishek.png")
    Abhishek_face_encoding = face_recognition.face_encodings(Abhishek_image)[0]

    Aishwarya_image = face_recognition.load_image_file("./images/Aishwarya.png")
    Aishwarya_face_encoding = face_recognition.face_encodings(Aishwarya_image)[0]

    Akhil_image = face_recognition.load_image_file("./images/Akhil.png")
    Akhil_face_encoding = face_recognition.face_encodings(Akhil_image)[0]

    Akhilesh_image = face_recognition.load_image_file("./images/Akhilesh.png")
    Akhilesh_face_encoding = face_recognition.face_encodings(Akhilesh_image)[0]

    Anil_image = face_recognition.load_image_file("./images/Anil.png")
    Anil_face_encoding = face_recognition.face_encodings(Anil_image)[0]

    AnkitE_image = face_recognition.load_image_file("./images/AnkitE.png")
    AnkitE_face_encoding = face_recognition.face_encodings(AnkitE_image)[0]

    AnkitP_image = face_recognition.load_image_file("./images/AnkitP.png")
    AnkitP_face_encoding = face_recognition.face_encodings(AnkitP_image)[0]

    aprajita_image = face_recognition.load_image_file("./images/aprajita.png")
    aprajita_face_encoding = face_recognition.face_encodings(aprajita_image)[0]

    Ayush_image = face_recognition.load_image_file("./images/Ayush.png")
    Ayush_face_encoding = face_recognition.face_encodings(Ayush_image)[0]

    Bhavesh_image = face_recognition.load_image_file("./images/Bhavesh.png")
    Bhavesh_face_encoding = face_recognition.face_encodings(Bhavesh_image)[0]

    Chitresh_image = face_recognition.load_image_file("./images/Chitresh.png")
    Chitresh_face_encoding = face_recognition.face_encodings(Chitresh_image)[0]

    Deepak_image = face_recognition.load_image_file("./images/Deepak.png")
    Deepak_face_encoding = face_recognition.face_encodings(Deepak_image)[0]

    Divyanshi_image = face_recognition.load_image_file("./images/Divyanshi.png")
    Divyanshi_face_encoding = face_recognition.face_encodings(Divyanshi_image)[0]

    Harsh_image = face_recognition.load_image_file("./images/Harsh.png")
    Harsh_face_encoding = face_recognition.face_encodings(Harsh_image)[0]

    Jighyasa_image = face_recognition.load_image_file("./images/Jighyasa.png")
    Jighyasa_face_encoding = face_recognition.face_encodings(Jighyasa_image)[0]

    Krishna_image = face_recognition.load_image_file("./images/Krishna.png")
    Krishna_face_encoding = face_recognition.face_encodings(Krishna_image)[0]

    Mitesh_image = face_recognition.load_image_file("./images/Mitesh.png")
    Mitesh_face_encoding = face_recognition.face_encodings(Mitesh_image)[0]

    Neha_image = face_recognition.load_image_file("./images/Neha.png")
    Neha_face_encoding = face_recognition.face_encodings(Neha_image)[0]

    Nisha_image = face_recognition.load_image_file("./images/Nisha.png")
    Nisha_face_encoding = face_recognition.face_encodings(Nisha_image)[0]

    Nitin_image = face_recognition.load_image_file("./images/Nitin.png")
    Nitin_face_encoding = face_recognition.face_encodings(Nitin_image)[0]

    Rahul_image = face_recognition.load_image_file("./images/Rahul.png")
    Rahul_face_encoding = face_recognition.face_encodings(Rahul_image)[0]

    Sahil_image = face_recognition.load_image_file("./images/Sahil.png")
    Sahil_face_encoding = face_recognition.face_encodings(Sahil_image)[0]

    Shikha_image = face_recognition.load_image_file("./images/Shikha.png")
    Shikha_face_encoding = face_recognition.face_encodings(Shikha_image)[0]

    Shivam_image = face_recognition.load_image_file("./images/Shivam.png")
    Shivam_face_encoding = face_recognition.face_encodings(Shivam_image)[0]

    Shivasha_image = face_recognition.load_image_file("./images/Shivasha.png")
    Shivasha_face_encoding = face_recognition.face_encodings(Shivasha_image)[0]

    Shriya_image = face_recognition.load_image_file("./images/Shriya.png")
    Shriya_face_encoding = face_recognition.face_encodings(Shriya_image)[0]

    ShriyaA_image = face_recognition.load_image_file("./images/ShriyaA.png")
    ShriyaA_face_encoding = face_recognition.face_encodings(ShriyaA_image)[0]

    Shruti_image = face_recognition.load_image_file("./images/Shruti.png")
    Shruti_face_encoding = face_recognition.face_encodings(Shruti_image)[0]

    ShubhamJ_image = face_recognition.load_image_file("./images/ShubhamJ.png")
    ShubhamJ_face_encoding = face_recognition.face_encodings(ShubhamJ_image)[0]

    Shubhankit_image = face_recognition.load_image_file("./images/Shubhankit.png")
    Shubhankit_face_encoding = face_recognition.face_encodings(Shubhankit_image)[0]

    Sonika_image = face_recognition.load_image_file("./images/Sonika.png")
    Sonika_face_encoding = face_recognition.face_encodings(Sonika_image)[0]

    Swati_image = face_recognition.load_image_file("./images/Swati.png")
    Swati_face_encoding = face_recognition.face_encodings(Swati_image)[0]

    Vikalp_image = face_recognition.load_image_file("./images/Vikalp.png")
    Vikalp_face_encoding = face_recognition.face_encodings(Vikalp_image)[0]

    Vinitha_image = face_recognition.load_image_file("./images/Vinitha.png")
    Vinitha_face_encoding = face_recognition.face_encodings(Vinitha_image)[0]

    Soumya_image = face_recognition.load_image_file("./images/Soumya.png")
    Soumya_face_encoding = face_recognition.face_encodings(Soumya_image)[0]

    Pushpraj_image = face_recognition.load_image_file("./images/Pushpraj.png")
    Pushpraj_face_encoding = face_recognition.face_encodings(Pushpraj_image)[0]

    Siddharth_image = face_recognition.load_image_file("./images/Siddharth.png")
    Siddharth_face_encoding = face_recognition.face_encodings(Siddharth_image)[0]

    Himanshu_image = face_recognition.load_image_file("./images/Himanshu.png")
    Himanshu_face_encoding = face_recognition.face_encodings(Himanshu_image)[0]

    Himanshu1_image = face_recognition.load_image_file("./images/Himanshu1.png")
    Himanshu_face_encoding1 = face_recognition.face_encodings(Himanshu1_image)[0]

    Himanshu2_image = face_recognition.load_image_file("./images/Himanshu2.png")
    Himanshu_face_encoding2 = face_recognition.face_encodings(Himanshu2_image)[0]

    Himanshu3_image = face_recognition.load_image_file("./images/Himanshu3.png")
    Himanshu_face_encoding3 = face_recognition.face_encodings(Himanshu3_image)[0]

    Himanshu4_image = face_recognition.load_image_file("./images/Himanshu4.png")
    Himanshu_face_encoding4 = face_recognition.face_encodings(Himanshu4_image)[0]


    known_face_encodings = [
        Gaurav_face_encoding,
        Mahima_face_encoding,
        Sardeep_face_encoding,
        Sandeep_face_encoding,
        Aakash_face_encoding,
        Abhishek_face_encoding,
        Aishwarya_face_encoding,
        Akhil_face_encoding,
        Akhilesh_face_encoding,
        Anil_face_encoding,
        AnkitE_face_encoding,
        AnkitP_face_encoding,
        aprajita_face_encoding,
        Ayush_face_encoding,
        Bhavesh_face_encoding,
        Chitresh_face_encoding,
        Deepak_face_encoding,
        Divyanshi_face_encoding,
        Harsh_face_encoding,
        Jighyasa_face_encoding,
        Krishna_face_encoding,
        Mitesh_face_encoding,
        Neha_face_encoding,
        Nisha_face_encoding,
        Nitin_face_encoding,
        Rahul_face_encoding,
        Sahil_face_encoding,
        Shikha_face_encoding,
        Shivam_face_encoding,
        Shivasha_face_encoding,
        Shriya_face_encoding,
        ShriyaA_face_encoding,
        Shruti_face_encoding,
        ShubhamJ_face_encoding,
        Shubhankit_face_encoding,
        Sonika_face_encoding,
        Swati_face_encoding,
        Vikalp_face_encoding,
        Vinitha_face_encoding,
        Soumya_face_encoding,
        Pushpraj_face_encoding,
        Siddharth_face_encoding,
        Himanshu_face_encoding,
        Himanshu_face_encoding1,
        Himanshu_face_encoding2,
        Himanshu_face_encoding3,
        Himanshu_face_encoding4,
        Pushpraj_face_encoding
    ]

    known_face_names = [
        "Gaurav Gaur",
        "Mahima",
        "Sardeep",
        "Sandeep",
        "Aakash",
        "Abhishek",
        "Aishwarya",
        "Akhil Verma",
        "Akhilesh",
        "Anil",
        "Ankit Engle",
        "Ankit Pawar",
        "Aparajita",
        "Ayush",
        "Bhavesh",
        "Chitresh",
        "Deepak",
        "Divyanshi",
        "Harsh",
        "Jighaysa",
        "Krishna",
        "Mitesh",
        "Neha",
        "Nisha",
        "Nitin",
        "Rahul",
        "Sahil",
        "Shikha",
        "Shivam",
        "Shivasha",
        "Shriya",
        "Shriya Agnihotri",
        "Shruti",
        "Shubham Joshi",
        "Shubhankit",
        "Sonika",
        "Swati",
        "Vikalp",
        "Vinitha",
        "Soumya",
        "Pushpraj",
        "Siddharth",
        "Himanshu",
        "Himanshu",
        "Himanshu",
        "Himanshu",
        "Himanshu",
        "Unknown"
    ]

    # if(val == 0):
    #     known_face_encodings.append(assign)
    #     known_face_names.append(name)
    
    List_Name={}
    Names={}
    face_locations = []
    face_encodings = []
    face_names = []
    lst = []
    process_this_frame = True
    try:
        connect = MongoClient()
        db= connect.Python
        print("Connected successfully!!!")
    except:
        print("Could not connect to MongoDB")
    video_capture.set(3,600)
    video_capture.set(4,400)
    root = tk.Tk()
    window_name = "window"
    
    # cap = cv.VideoCapture(0)
    
    # cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    objects=set()
    whT = 320
    confThreshold =0.2
    nmsThreshold= 0.2

    #### LOAD MODEL
    ## Coco Names
    classesFile = "coco.names"
    classNames = []
    with open(classesFile, 'r') as f:
        classNames = f.read().rstrip('\n').split("\n")
    print(classNames)
    ## Model Files
    modelConfiguration = "yolov3-320.cfg"
    modelWeights = "yolov3-320.weights"
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def findObjects(outputs,img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        global object
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
                    # print(classId)
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        for i in indices:
            i = i
            try:
                if(i<len(classIds) and i<len(confs) and i<len(bbox) ):
                    box = bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    # print(x,y,w,h) 
                    cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
                    cv.putText(img,f'{classNames[classIds[i]].upper()} ', 
                            (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    objects.add(classNames[classIds[i]].upper())
                    # popDb.addObjects(dict({"objectName":classNames[classIds[i]].upper()}))
                    # print(classNames[classIds[i]].upper())  
                    
            except:
                print('exception occured')
        print(objects)   

    root = tk.Tk()

    while True:
       
        # success, img = cap.read()
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) 


        width = root.winfo_screenwidth() #1500
        height = root.winfo_screenheight()- 100 #1080
        dim = (width, height)
        # success, img = cap.read()       


        ret, frame = video_capture.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            i=len(face_encodings)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            unknownCount=0

            for face_encoding in face_encodings:
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print(face_distances)
                flag = False
                for face_distance in face_distances :
                    if face_distance < 0.5 :
                        flag = True
                        #best_match_index= len(known_face_names)-1  
                        #best_match_index = np.argmin(face_distances)
                if(flag):
                    best_match_index = np.argmin(face_distances)
                else:
                    best_match_index= len(known_face_names)-1 

                if matches[best_match_index] is None :
                        unknownCount=unknownCount+1
                if matches[best_match_index]:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    name = known_face_names[best_match_index]
                    if(name not in  List_Name.keys()) :
                            list = []
                            list.append(now.strftime("%H:%M:%S"))
                            # s = s+str(name)+" : "+str(list)+"\n"
                            # print(s)
                            List_Name[name]= list
                            
                    else:
                        timelist = List_Name[name]
                        lastentry = timelist[-1]
                        lm1 = int(lastentry[3])
                        lm2 = int(lastentry[4])
                        lastmin = lm1+lm2
                        cm1 = int(current_time[3])
                        cm2 = int(current_time[4])
                        currmin = cm1+cm2
                        diff = currmin - lastmin
                        if(diff > 1):
                            List_Name[name].append(current_time)
                    
                face_names.append(name)
        for name in List_Name.keys():
            for value in List_Name.values() :
                document={
                    "name" :str(name),
                    "time":str(value)
                }
            db = connect.Python
            collection = db.Attendence
            collection.insert_one(document)
        process_this_frame = not process_this_frame
        lenth=len(List_Name.keys())
        totalFace = "Total face : "+str(lenth)
        liveface = "Live Face : "+str(i)
        font = cv2.FONT_HERSHEY_COMPLEX
        # x,y,w,h = 0,400,200,100
        # cv2.rectangle(frame, (0,300), (x + w, y + h), (0,0,0), -1)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
            cv2.rectangle(frame, (left, bottom ), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            roi_color = frame[top : top + bottom ,left :left +right]
            print("[INFO] Object found. Saving locally.") 
            cv2.imwrite(name + '_faces.jpg', roi_color)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  
            # popDb.addAppearence(dict({"name":name}))
                
        cv2.putText(frame, "Analytics :", (20, 30), font, 1, (0, 0, 250), 2)
        cv2.putText(frame, liveface, (20, root.winfo_screenheight()-150), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, totalFace, (20, root.winfo_screenheight()-110), font, 1, (0, 0, 0), 2)
        a = 0
        for (label, count) in enumerate(List_Name.keys()): 
            cv2.putText(frame, f"{label+1}: {count}", (30, 60+a), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            a=a+20
        # cv2.imshow('Video', frame)    
        width = root.winfo_screenwidth() #1500
        height = root.winfo_screenheight()- 100 #1080
        dim = (width, height)
        # success, frame = video_capture.read()     
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,frame)
 
        cv2.imshow('Image', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    Data={}
    names = List_Name.keys()
    # collection.insert_one(List_Name)
#     return {
#         'data': json.loads(json_util.dumps(List_Name))
#     }

# @app.after_request
# def after_request(response):
#   response.headers.set('Access-Control-Allow-Origin', '*')
#   response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#   response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#   return response

# @app.route("/sentiment",methods = ['GET'])
def live_face_count():
    # Color RGB Codes & Font
    WHITE_COLOR = (255, 255, 255)
    GREEN_COLOR = (0, 255, 0)
    BLUE_COLOR = (255, 255, 104)
    FONT = cv2.QT_FONT_NORMAL

    # Frame Width & Height
    FRAME_WIDTH = 6400
    FRAME_HEIGHT = 4900


    class BoundingBox:
        def __init__(self, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h

        @property
        def origin(self) -> tuple:
            return self.x, self.y

        @property
        def top_right(self) -> int:
            return self.x + self.w

        @property
        def bottom_left(self) -> int:
            return self.y + self.h


    def draw_face_rectangle(bb: BoundingBox, img, color=BLUE_COLOR):
        cv2.rectangle(img, bb.origin, (bb.top_right, bb.bottom_left), color, 2)


    def draw_landmark_points(points: np.ndarray, img, color=WHITE_COLOR):
        if points is None:
            return None
        for (x, y) in points:
            cv2.circle(img, (x, y), 1, color, -1)


    def write_label(x: int, y: int, label: str, img, color=BLUE_COLOR):
        if label == NO_FACE_LABEL:
            cv2.putText(img, label.upper(), (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)), FONT, 1, color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + 10, y - 10), FONT, 1, color, 2, cv2.LINE_AA)
        # if label != NO_FACE_LABEL:
        #     popDb.addSentiments(dict({"sentimentName":label.capitalize()}))

    li = []
    class RealTimeEmotionDetector:
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        vidCapture = None

        def __init__(self, classifier_model: ImageClassifier):
            self.__init_video_capture(camera_idx=0, frame_w=FRAME_WIDTH, frame_h=FRAME_HEIGHT)
            self.classifier = classifier_model

        def __init_video_capture(self, camera_idx: int, frame_w: int, frame_h: int):
            self.vidCapture = cv2.VideoCapture(camera_idx)
            self.vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
            self.vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

        def read_frame(self) -> np.ndarray:
            rect, frame = self.vidCapture.read()
            return frame

        def transform_img(self, img: np.ndarray) -> np.ndarray:
            # load the input image, resize it, and convert it to gray-scale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray-scale
            resized_img = self.CLAHE.apply(gray_img)  # resize
            return resized_img

        def execute(self, wait_key_delay=5, quit_key='q', frame_period_s=0.75):
            frame_cnt = 0
            predicted_labels = ''
            old_txt = None
            rectangles = [(0, 0, 0, 0)]
            landmark_points_list = [[(0, 0)]]
            while cv2.waitKey(delay=wait_key_delay) != ord(quit_key):
                frame_cnt += 1

                frame = self.read_frame()
                if frame_cnt % (frame_period_s * 100) == 0:
                    frame_cnt = 0
                    predicted_labels = self.classifier.classify(img=self.transform_img(img=frame))
                    rectangles = self.classifier.extract_face_rectangle(img=frame)
                    landmark_points_list = self.classifier.extract_landmark_points(img=frame)
                for lbl, rectangle, lm_points in zip(predicted_labels, rectangles, landmark_points_list):
                    draw_face_rectangle(BoundingBox(*rectangle), frame)
                    draw_landmark_points(points=lm_points, img=frame)
                    write_label(rectangle[0], rectangle[1], label=lbl, img=frame)

                    if old_txt != predicted_labels:
                        print('[INFO] Predicted Labels:', predicted_labels)
                        li.append(predicted_labels)
                        old_txt = predicted_labels

                cv2.imshow('Emotion Detection', frame)

            cv2.destroyAllWindows()
            self.vidCapture.release()


    def run_real_time_emotion_detector(
            classifier_algorithm: str,
            predictor_path: str,
            dataset_csv: str,
            dataset_images_dir: str = None):
        from utils.data_land_marker import LandMarker
        from utils.image_classifier import ImageClassifier
        from os.path import isfile

        land_marker = LandMarker(landmark_predictor_path=predictor_path)

        if not isfile(dataset_csv):  # If data-set not built before.
            print('[INFO]', f'Dataset file: "{dataset_csv}" could not found.')
            from data_preparer import run_data_preparer
            run_data_preparer(land_marker, dataset_images_dir, dataset_csv)
        else:
            print('[INFO]', f'Dataset file: "{dataset_csv}" found.')

        classifier = ImageClassifier(csv_path=dataset_csv, algorithm=classifier_algorithm, land_marker=land_marker)
        print('[INFO] Opening camera, press "q" to exit..')
        RealTimeEmotionDetector(classifier_model=classifier).execute()


    if __name__ == "__main__":
        """The value of the parameters can change depending on the case."""
        run_real_time_emotion_detector(
            classifier_algorithm='RandomForest',  # Alternatively 'SVM'.
            predictor_path='utils/shape_predictor_68_face_landmarks.dat',
            dataset_csv='data/csv/dataset.csv',
            dataset_images_dir='data/raw'
        )
        print('Successfully terminated.')
        # return {
#             'Data': li
#             }

# @app.after_request
# def after_request(response):
#     response.headers.set('Access-Control-Allow-Origin', '*')
#     response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

# @app.route("/objectdetection")
# def live_object_count():

    # window_name = "window"
    
    # # cap = cv.VideoCapture(0)
    
    # # cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    # objects=set()
    # whT = 320
    # confThreshold =0.2
    # nmsThreshold= 0.2

    # #### LOAD MODEL
    # ## Coco Names
    # classesFile = "coco.names"
    # classNames = []
    # with open(classesFile, 'r') as f:
    #     classNames = f.read().rstrip('\n').split("\n")
    # print(classNames)
    # ## Model Files
    # modelConfiguration = "yolov3-320.cfg"
    # modelWeights = "yolov3-320.weights"
    # net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # def findObjects(outputs,img):
    #     hT, wT, cT = img.shape
    #     bbox = []
    #     classIds = []
    #     confs = []
    #     global object
    #     for output in outputs:
    #         for det in output:
    #             scores = det[5:]
    #             classId = np.argmax(scores)
    #             confidence = scores[classId]
    #             if confidence > confThreshold:
    #                 w,h = int(det[2]*wT) , int(det[3]*hT)
    #                 x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
    #                 bbox.append([x,y,w,h])
    #                 classIds.append(classId)
    #                 confs.append(float(confidence))
    #                 # print(classId)
    #     indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    #     for i in indices:
    #         i = i
    #         try:
    #             if(i<len(classIds) and i<len(confs) and i<len(bbox) ):
    #                 box = bbox[i]
    #                 x, y, w, h = box[0], box[1], box[2], box[3]
    #                 # print(x,y,w,h) 
    #                 cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
    #                 cv.putText(img,f'{classNames[classIds[i]].upper()} ',
    #                         (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    #                 objects.add(classNames[classIds[i]].upper())
    #                 # popDb.addObjects(dict({"objectName":classNames[classIds[i]].upper()}))
    #                 # print(classNames[classIds[i]].upper())  
                    
    #         except:
    #             print('exception occured')
    #     print(objects)   

    # root = tk.Tk()

    # # while True:
    # #     width = root.winfo_screenwidth() #1500
    # #     height = root.winfo_screenheight()- 100 #1080
    # #     dim = (width, height)
    # #     success, img = video_capture.read()
    # #     img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # #     blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    # #     net.setInput(blob)
    # #     layersNames = net.getLayerNames()
    # #     outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # #     outputs = net.forward(outputNames)
    # #     findObjects(outputs,img)

    # #     cv2.imshow('Image', img)
    # #     if cv2.waitKey(1) & 0xFF == ord('q'):
    # #         break
    return {
        # 'Total_object': jsonify(objects)
        'Data' : json.loads(json_util.dumps(objects))
    }

@app.after_request
def after_request(response):
    response.headers.set('Access-Control-Allow-Origin','*')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(port=5000,debug=True)