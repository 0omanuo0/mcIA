from logging import exception
from time import time
import pyautogui
import cv2
import numpy as np
import timeit
import mss
import prueba
from PIL import Image

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1024, 1024)
monitor = {'top': 40, 'left': 0, 'width': 1024, 'height': 1024}
mss1 = mss.mss()
m1=prueba.SCNN()

def getWindow():
    data = np.array([])
    while True:
        t1 = timeit.default_timer()
        #frame = pyautogui.screenshot(region=(0, 0, 300, 400))
        frame = mss1.grab(monitor)
        input = np.array(frame)
        frame = Image.fromarray(input).convert('RGB').resize((512,512))
        t2 = timeit.default_timer()
        out = m1.segment(frame)
        t3 = timeit.default_timer()
        out = np.array(out.convert('RGB'))
        try:
            cv2.imshow('Live',out)
        except:
            pass
        t4 = timeit.default_timer()
        if int(t2) - int(t1) == 1:   
            data = np.append(data, 1/(t3-t1))
            print('time(ms): ', t2-t1,t3-t2,t4-t3,t4-t1, '\nfps: ', 1/(t3-t1), '\nmean: ', np.mean(data))
        # Stop recording when we press 'q'
        if cv2.waitKey(1) == ord('q'):
            break
                
  
# Destroy all windows
if __name__ == '__main__':
    getWindow()
    cv2.destroyAllWindows()