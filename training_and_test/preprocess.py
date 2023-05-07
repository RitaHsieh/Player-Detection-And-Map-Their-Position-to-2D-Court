# data preprocessing

from tensorflow.python.ops.math_ops import segment_max
import os, cv2
import numpy as np

list = [0, 40, 80, 120, 160, 200]
print(list)

IMG_WIDTH=512
IMG_HEIGHT=288

def pre_process(img_folder, mask_folder, n_class, dataset_size):
   
    img_data=[]
    img_mask=[]
    for dir1 in os.listdir(img_folder):
        # print(dir1)
        if dir1==".DS_Store":
            continue
        image_path= os.path.join(img_folder, dir1)
        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        img_data.append(image)

        mask_path= os.path.join(mask_folder, dir1[:-4] + ".npy")
        image_mask = np.load(mask_path)
        seg_labels = np.zeros((image_mask.shape[0], image_mask.shape[1] ,n_class))
        
        for c in range(3,6):
            seg_labels[:, :, c-3] = (image_mask == list[c]).astype(int)


        seg_labels[:,:,3] = (image_mask < 110).astype(int)
        seg_lab = np.zeros((IMG_HEIGHT, IMG_WIDTH ,  n_class))

        for s in range(n_class):
          seg_lab[:,:,s] = cv2.resize(seg_labels[:,:,s], (IMG_WIDTH, IMG_HEIGHT))

        img_mask.append(seg_lab)
        
    print(np.array(img_mask).shape)
    img_data, img_mask = np.array(img_data), np.array(img_mask).reshape((dataset_size, IMG_WIDTH * IMG_HEIGHT , n_class))
    return img_data, img_mask

# extract the image array and class name
# extract the image array and class name

# the path must be set before using create_dataset
img_data, mask = pre_process(r'images', r'images_mask', n_class=4, dataset_size=865)
img_data_valid, mask_valid = pre_process(r'test_images', r'test_images_mask', n_class=4, dataset_size=100)


np.save("img_data_288512.npy", img_data)
np.save("mask_288512.npy", mask)
np.save("img_data_valid_288512.npy", img_data_valid)
np.save("mask_valid_288512.npy", mask_valid)