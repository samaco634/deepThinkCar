import cv2
import time

stop_sign = cv2.CascadeClassifier('./models/cascade_stop_sign.xml')
stop_width, stop_height = 40, 40


# real driving routine
def isStopSignDetected(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    
    if len(stop_sign_scaled) == 0 :
        isStop = False
    else:
        # Detect the stop sign, x,y = origin points, w = width, h = height
        for (x, y, w, h) in stop_sign_scaled:
            # Draw rectangle around the stop sign
            stop_sign_rectangle = cv2.rectangle(img, (x,y),
                                                (x+w, y+h),
                                                (0, 255, 0), 3)
            # Write "Stop sign" on the bottom of the rectangle
            stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                         text="Stop Sign",
                                         org=(x, y+h+30),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=(0, 0, 255),
                                         thickness=2, lineType=cv2.LINE_4)
            if w > stop_width or h > stop_height:
                isStop = True
    return isStop, img


if __name__ == '__main__':
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()
    
    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    
    camera = cv2.VideoCapture(-1)
    camera.set(3, 320)
    camera.set(4, 240)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
    
        ret, img = camera.read()
        counter += 1
        
        if ret:
            isStop, img = isStopSignDetected(img)
            
                            # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
              end_time = time.time()
              fps = fps_avg_frame_count / (end_time - start_time)
              start_time = time.time()

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            text_location = (left_margin, row_size)
            cv2.putText(img, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)
            
            cv2.imshow("stop sign detect", img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
