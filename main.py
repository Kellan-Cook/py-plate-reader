import cv2
import cv2.data

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
car_cascade_01 = cv2.CascadeClassifier("./cars1.xml")
plate_cascade_01 = cv2.CascadeClassifier("./rus_plates.xml")
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    rval, frame = vc.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    car_data_1 = car_cascade_01.detectMultiScale(img_gray, scaleFactor=1.1)
    plate_data1 = plate_cascade_01.detectMultiScale(img_gray, scaleFactor=1.1)
    for x, y, w, h in car_data_1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for x, y, w, h in plate_data1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow("preview")
vc.release()
