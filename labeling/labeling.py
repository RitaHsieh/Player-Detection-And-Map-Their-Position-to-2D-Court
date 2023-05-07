import cv2
import numpy as np
import os
import shutil

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(data['img'], (x,y), 3, (0,0,255), 5, 16) 

        # 改變顯示 window 的內容
        cv2.imshow("Image", data['img'])
        
        # 顯示 (x,y) 並儲存到 list中
        print("get points: (x, y) = ({}, {})".format(x, y))
        data['points'].append((x,y))

def get_points(im):
    # 建立 data dict, img:存放圖片, points:存放點
    data = {}
    data['img'] = im.copy()
    data['points'] = []
    
    # 建立一個 window
    cv2.namedWindow("Image", 0)
    
    # 改變 window 成為適當圖片大小
    h, w, dim = im.shape
    print("Img height, width: ({}, {})".format(h, w))
    cv2.resizeWindow("Image", w, h)
        
    # 顯示圖片在 window 中
    cv2.imshow('Image',im)
    
    # 利用滑鼠回傳值，資料皆保存於 data dict中
    cv2.setMouseCallback("Image", mouse_handler, data)
    
    # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 回傳點 list
    return data['points']

def transform(pts1, pts2, mask):
    # pts1 : to
    # pts2 : from 
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv2.getPerspectiveTransform(pts2,pts1)

    dst = cv2.warpPerspective(mask,M,(640, 360))

    dst = dst[:,:,0]
    
    dst = np.where(abs(dst-40) <= 15, 40, dst)
    dst = np.where(abs(dst-80) <= 15, 80, dst)
    dst = np.where(abs(dst-120) <= 15, 120, dst)
    dst = np.where(abs(dst-160) <= 15, 160, dst)
    dst = np.where(abs(dst-200) <= 15, 200, dst)
    dst = np.where((dst%40)!=0, 0, dst)

    return dst

mask = cv2.imread(r"labeling/map_grey.png")
print(mask.shape)
path=r"labeling/images_tolabel"  # 待標註的圖
savepath=r"labeling/images_mask" # 2D球場
new_path=r"labeling/images"      # 標註後存放的位置
annotationpath=r"labeling/images_annotation"
print("Click on the screen and press any key for end process")

# transform_base.npy 存放球場在2D平面的點
transform_base = np.load(r"labeling/transform_base.npy")
# print(transform_base)
rec = []
for i in os.listdir(path):

    image_path= os.path.join(path, i)
    if image_path[-1] == 'g':

        # img = cv2.imread(image_path)
        # img = np.array(img, np.float32)
        # img = img/255.0
        # print(img)
        # obj_path = os.path.join(annotationpath, i)
        # cv2.imwrite(obj_path, img)

        print(image_path)
        img = cv2.imread(image_path)
        # print(img.shape)
        img = cv2.resize(img, [640, 360])
        pts = get_points(img)
        pts_type = int(input("types: "))
        # print(transform_base[pts_type])
        if pts_type == 5:
            rec.append(i)
        else:
            img_mask = transform(pts,transform_base[pts_type],mask)
            obj_path = os.path.join(new_path, i)
            obj_path2 = os.path.join(savepath, i[:-4] + ".npy")
            # print(img_mask.dtype)
            # print(np.argwhere(img_mask%40!=0))
            # cv2.imwrite(obj_path, img_mask)
            np.save(obj_path2, img_mask)
            shutil.move(image_path, obj_path)
            

print(rec)

for f in rec:
    rm_path= os.path.join(path, f)
    os.remove(rm_path)
