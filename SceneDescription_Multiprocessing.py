import cv2 as cv
import time
import pyttsx3 
import multiprocessing
import math

def SpeakObject(q):
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 220)

    while q.empty() is False:
        obj, dist = q.get()
        tts_engine.say(str(obj) + " ahead at " + str(dist) + " meters")
    
    tts_engine.runAndWait()
    del tts_engine

def getDistance(real_width, width_in_frame):
    #Camera: DJI Osmo Pocket
    #f = 26mm / 12mm , ImgWidth = 4000 pixels, Sensor = 1/2.3" => 6.17 mm width
    #f(pix) = f(mm) * ImgWidth / SensorWidth
    # Person Real Width = 0.385 m, Car Real Width = 4 m, Bus = 10m, Truck = 5m

    # distance = (real_width * 16855.7536) / width_in_frame #f = 26mm
    distance = (real_width * 7779.5786) / width_in_frame #f = 12mm
    return distance

def WebCam_Video(q):
    Conf_threshold = 0.8
    NMS_threshold = 0.8
    COLORS = [(0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    class_name = []
    with open('classnames.txt', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    SD_Model = cv.dnn_DetectionModel(net)
    SD_Model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    cap = cv.VideoCapture("LondonWalkDay.mov")
    starting_time = time.time()
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if ret == False:
            break
        classes, scores, boxes = SD_Model.detect(frame, Conf_threshold, NMS_threshold)
        if len(classes) != 0:
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                score = round(score * 100, 2)
                if class_name[classid] == 'car':
                    real_width = 4
                elif class_name[classid] == 'person':
                    real_width = 0.385
                elif class_name[classid] == 'truck':
                    real_width = 4
                elif class_name[classid] == 'bus':
                    real_width = 10
                dist = round(getDistance(real_width, box[2]), 2)
                label = "%s: %f" % (class_name[classid].upper(), dist)
                cv.rectangle(frame, box, color, 1)
                cv.putText(frame, label, (box[0] + 10, box[1]+30), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                q.put((class_name[classid], math.floor(dist)))
        else:
            pass

        endingTime = time.time() - starting_time
        fps = frame_counter/endingTime
        cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    q = multiprocessing.Queue()

    videoProcess = multiprocessing.Process(target=WebCam_Video, args=(q,))
    ttsProcess = multiprocessing.Process(target=SpeakObject, args=(q,))

    videoProcess.start()
    ttsProcess.start()

    videoProcess.join()
    ttsProcess.join()