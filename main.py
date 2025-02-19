from ultralytics import YOLO
import cv2
import easyocr
from collections import Counter, deque
import pytesseract
import time
#model = YOLO('./models/best.pt')         
model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/best.pt')
cap = cv2.VideoCapture(0)

image = False

ret = True
vehicles = [2,3,5,7]
detected = deque()
reader = easyocr.Reader(["en"])
most_common = "waiting"
license_plate_number_list = deque()
x1 = 0
y1 = 0
while ret:
    ret, frame = cap.read()
    if ret or image:
        if image:
            frame = cv2.imread('./Test_Video/image.png')
            #print(frame)
        # Detecting License Plate
        results = license_plate_detector(frame)[0]
        #print(f"boxes are: {results.boxes}")
        if results.boxes is None:
            pass
        else:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                license_plate = frame[int(y1):int(y2), int(x1):int(x2), :]
                cv2.imshow("crop", license_plate)

                # After cropping license plate from image


                license_plate_grey = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_grey, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                #cv2.imshow("thresh", license_plate_thresh)
                #plate_number = pytesseract.image_to_string(license_plate_thresh, config="--psm 7")
                #print(f"result is out:  {result}") 
                #print(f"pytesseract: {plate_number}")

            
                numbers = reader.readtext(license_plate_thresh)
                cv2.imshow("thresh", license_plate_thresh)
                print(f"numbers is: {numbers}")
                texts = []
                texts.clear()
                largest_text_width = 0    
                license_plate_number = "waiting"
                if numbers:
                    for number in numbers:
                        #print(f"number is: {number}")
                        bbox, text = number[0], number[1]
                        print(f"text detected is: text: {text}")
                        top_left = (int(bbox[0][0]), int(bbox[0][1]))
                        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
                        #print(f"type: {type(top_left)}, {type(bottom_right)}")
                        cv2.rectangle(license_plate, top_left, bottom_right, (0,255,0), 3 )
                        cv2.putText(license_plate, text, bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        text_width = bbox[2][0] - bbox[0][0]
                        print(f"text_width is: {text_width}")
                        if (text_width) > largest_text_width:
                            largest_text_width = text_width
                            license_plate_number = text
                        detected.append(text)
                        texts.append(text)
                        #print(f"detected list: {detected}")
                        #cv2.putText(frame, text, (int(x1), int(y1 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                #print(f"easy ocr: text:{text}, conf: {conf}")
                cv2.imshow("plate",license_plate)
                license_plate_number_list.append(license_plate_number)
                print(f"License Plate Detected Result is: {license_plate_number}")
            boxes = results.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
            confs = results.boxes.conf  # Confidence scores
            class_ids = results.boxes.cls  # Class IDs
            #for box, conf, class_id in zip(boxes, confs, class_ids):
            #    x1, y1, x2, y2 = map(int, box)  # Convert to integers
            #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #    label = f"ID: {int(class_id)} {conf:.2f}"
            #    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = results.plot()
            if len(license_plate_number_list) > 8:
                counter = Counter(license_plate_number_list)
                most_common = counter.most_common(1)
                #print(f"before pop: {detected}")
                delete_num = len(license_plate_number_list) - 10
                for index in range(delete_num):             # make sure the length of DQ is 10
                    license_plate_number_list.popleft()             
                #print(f"after pop: {detected}")
            print(f"most common result is: {most_common}")
            
            cv2.putText(frame, most_common[0][0], (int(x1), int(y1 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


            
            if frame is None:
                "Error!!!!!! image is missing"
            else:
                cv2.imshow("YOLOv8 Detection", frame)
        
        if cv2.waitKey(1) | 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
print(detected)




