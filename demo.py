import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import math
import keras.backend as K
from model import segnet
model_test = segnet((288, 512, 3), 4)
model_test.load_weights('segnet_09844_09811.h5')
def count_dot(a):
    count = 0
    for i in a:
        if len(i) != 0:
            count += 1

    return count

def court(img):
    
    h, w = img.shape[:2]
    # img_with_hull = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    if np.max(img) == 0:
        return None #, img_with_hull
    else:
        print(np.max(img))
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 100:
        return None #, img_with_hull
    hull = cv2.convexHull(cnt)
    arr = np.zeros((h, w), np.uint8)
    box = cv2.approxPolyDP(hull, 3, True)
    
    # length = len(hull)
    # for i in range(len(hull)):
    #     img_with_hull = cv2.line(img_with_hull, 
    #                         tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0, 120, 120), 2)
    
    minimum = box[:,0,:].argmin(axis=0) # x_min, y_min
    maximum = box[:,0,:].argmax(axis=0) # x_max, y_max
    
    # get the dot
    left = box[minimum[0]][0]
    right = box[maximum[0]][0]
    top = box[minimum[1]][0]
    bottom = box[maximum[1]][0]

    # if the dot near edge, delete it
    
    if left[0] == 0:
        left = []
    if right[0] == w-1:
        right = []
    if top[0] == 0 or top[0] == w-1:
        top = []
    if bottom[0] == 0 or bottom[0] == w-1:
        bottom = []

    ans = [top, right, bottom, left]
    return ans #, img_with_hull
    
def transform(pts1, pts2, mask):
    # pts1 : to
    # pts2 : from
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv2.getPerspectiveTransform(pts2,pts1)
    dst = cv2.warpPerspective(mask,M,(1224, 690))

    return M, dst
    

# path = "result/whole_court.mp4" #result/whole_court.mp4" #"result/guest_setplay.mp4"
kernel = np.ones((3,3), np.uint8)
count = 0
mask = cv2.imread("200_255.png", cv2.COLOR_BGR2RGB)
# mask = cv2.imread("unused_img/200_255.png", cv2.COLOR_BGR2RGB)
print(mask.shape)
ob = np.load("transform_base_back.npy")
note = np.load("result/result_whole_court.npy")#result/result_whole_court.npy") #
video = "result/whole_court.mp4"


cap = cv2.VideoCapture(video)

while(True):
  # 從攝影機擷取一張影像
  
  ret, frame = cap.read()

  if ret:
    player = note[count]
    count += 1
    mask_dis = mask.copy()
    mask_dis = cv2.cvtColor(mask_dis, cv2.COLOR_GRAY2BGR)
    
    image = cv2.resize(frame, (512, 288))
    img = np.array(image, np.float32)
    img /= 255 
    img = [img]
    ans = model_test.predict(np.array(img))
    ans = ans.reshape((1, 288, 512, 4))
    handle_img = ans[0]
    handle_img = np.where(handle_img>0.5, 255, 0)
    handle_img = np.array(handle_img, np.uint8)
    
    # 顯示圖片
    for i in range(4):
        handle_img[:,:,i] = cv2.dilate(handle_img[:,:, i], kernel, iterations = 3)
        handle_img[:,:,i] = cv2.erode(handle_img[:,:, i], kernel, iterations = 3)
        
    ## Find the four charecteristic dot of the court  
    left_dot = 0
    right_dot = 0
    center_dot = 0
    pts1=[]
    pts2=[]

    left = court(handle_img[:,:,0])
    right = court(handle_img[:,:,1])
    center = court(handle_img[:,:,2])

    # left, show1 = court(handle_img[:,:,0])
    # right, show2 = court(handle_img[:,:,1])
    # center, show3 = court(handle_img[:,:,2])
    
    if left is not None:
        left_dot = count_dot(left)
    if right is not None:
        right_dot = count_dot(right)
    if center is not None:
        center_dot = count_dot(center)

    if left_dot == 4:
        pts1 = ob[0]
        pts2 = left
    elif right_dot == 4:
        pts1 = ob[2]
        pts2 = right
    elif center_dot == 4 and right_dot < 2 and left_dot < 2:
        pts1 = ob[1]
        pts2 = center
        temp = pts2[1]
        pts2[1] = pts2[3]
        pts2[3] = temp

    elif right_dot == 0 :
        print("right dot == 0")
        pts1 = ob[3]
        temp_list = [center[0], center[2], left[2], left[1]]
        if count_dot(temp_list) < 4:
            continue
        pts2 = [center[0], center[2], left[2], left[1]]
    elif left_dot == 0:
        print("left dot == 0")
        pts1 = ob[4]
        temp_list = [right[3], right[2], center[2], center[0]]
        if count_dot(temp_list) < 4:
            continue
        pts2 = [right[3], right[2], center[2], center[0]]


    # for i in [show1, show2, show3]:
    #     for j in range(4):
    #         i = cv2.circle(i,
    #                         (pts2[j]), radius=3, color=(0,255,0), thickness=5)
            
   
    # show = np.hstack((show1, show3, show2))
    # show = cv2.resize(show, (1277, 240))

    ## get perspective matrix
    T1, dst = transform(pts1, pts2, image)
    
    ## Transform the player to 2D
    # pts1 : to
    # pts2 : from  
    for k in pts2:
        position = T1.dot([[k[0]], [k[1]], [1]])
        x = int(position[0]/position[2])
        y = int(position[1]/position[2])
        print("p1: ",x, y)
        cv2.circle(mask_dis, (x, y), 10, (0, 180, 180), 10)

    for i in player[0]:
        x_or = float(i[0]/2.5)
        y_or = float(i[1]/2.5)
        matrix = np.array([[x_or], [y_or], [1.0]], dtype=np.float32)
        position = T1.dot(matrix)
        
        if position[2] == 0:
            break
        x = int(position[0]/position[2])
        y = int(position[1]/position[2])
        cv2.circle(image, (int(x_or), int(y_or)), 5, (255, 255, 255), 10)
        cv2.circle(mask_dis, (x, y), 5, (255, 255, 255), 15)
    
    for j in player[1]:
        x_or = float(j[0]/2.5)
        y_or = float(j[1]/2.5)
        matrix = np.array([[x_or], [y_or], [1.0]], dtype=np.float32)
        position = T1.dot(matrix)
        
        if position[2] == 0:
            break
        x = int(position[0]/position[2])
        y = int(position[1]/position[2])
        cv2.circle(image, (int(x_or), int(y_or)), 5, (0, 120, 255), 10)
        cv2.circle(mask_dis, (x, y), 5, (0, 120, 255), 15)
    
    
    image = cv2.resize(image, (420, 240))
    mask_dis = cv2.resize(mask_dis, (437, 240))
    dst = cv2.resize(dst, (420, 240))
    print(mask_dis.shape, image.shape)
    display = np.hstack((mask_dis, image, dst))
    # display = np.vstack((display, show))
    
    cv2.imshow('frame', display)

    # if cv2.waitKey(0) & 0xFF == ord('k'):
    #     continue

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

