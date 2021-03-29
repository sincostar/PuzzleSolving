import cv2
import math
import random
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


# sigma to change the blight and dark rate
sigma1 = 0.3
sigma2 = 0.32
sigma3 = 0.33
sigma4 = 0.33

RANDOM_SEED = 2
WHOLE_IMG_NUM = 10000
WHOLE_IMG_SEED = []

File_path = r'F:\Deepzzle\PuzzleSolving'
Coco_path = r"F:\COCO 2014\train2014"

dis_dic = {"01": 1., "02": 2., "03": 1., "04": 1.4142, "05": 2.2361, "06": 2., "07": 2.2361, "08": 2.8284,
           "12": 1., "13": 1.4142, "14": 1., "15": 1.4142, "16": 2.2361, "17": 2., "18": 2.2361,
           "23": 2.2361, "24": 1.4142, "25": 1, "26": 2.8284, "27": 2.2361, "28": 2.,
           "34": 1., "35": 2., "36": 1., "37": 1.4142, "38": 2.2361,
           "45": 1., "46": 1.4142, "47": 1., "48": 1.4142,
           "56": 2.2361, "57": 1.4142, "58": 1.,
           "67": 1., "68": 2.,
           "78": 1.}


def get_shuffle_image(img_arr, cur_id=0, save_path=None):
    """

    :param img_arr:
    :param cur_id:
    :param save_path:
    :return:  the shuffle image (ndarray) and shuffle label.

    Get the shuffle image for training and testing.
    If save image path is Not None:
        Save the image as .jpg file and file name will include the image id and shuffle label.
    """

    central_fragment = img_arr.pop(4)
    change_img_arr = img_arr.copy()
    # create 4 more blight rate fragment
    # random.seed(RANDOM_SEED)
    # rand_id1, rand_id2, rand_id3, rand_id4, rand_id5, rand_id6 = random.sample(range(8), 6)
    # img_list[rand_id1] = (img_list[rand_id1] * (1-sigma1) + 255 * sigma1).astype(np.uint8)
    # img_list[rand_id2] = (img_list[rand_id2] * (1-sigma2) + 255 * sigma2).astype(np.uint8)
    # img_list[rand_id1] = (img_list[rand_id1] * (1-sigma1) + 255 * sigma1).astype(np.uint8)
    # img_list[rand_id2] = (img_list[rand_id2] * (1-sigma2) + 255 * sigma2).astype(np.uint8)

    # create 2 more dark rate fragment
    # img_list[rand_id5] = (img_list[rand_id5] * (1 - sigma1) + 0 * sigma1).astype(np.uint8)
    # img_list[rand_id6] = (img_list[rand_id6] * (1 - sigma2) + 0 * sigma2).astype(np.uint8)
    # new_img1 = np.concatenate((img_list[0], img_list[1], img_list[2], img_list[3]), axis=1)
    # new_img2 = np.concatenate((img_list[4], img_list[5], img_list[6], img_list[7]), axis=1)
    # new_img = np.concatenate((new_img1, new_img2), axis=0)
    # cv2.imshow('different rate', new_img)

    # shuffle the fragment
    random.seed(WHOLE_IMG_SEED[cur_id])
    shuffle_list = list(range(8))
    random.shuffle(shuffle_list)
    p_label = []
    for i in range(8):
        cur_index = int(shuffle_list[i])
        cur_p_label = [0] * 8
        cur_p_label[cur_index] = 1
        change_img_arr[i] = img_arr[cur_index]
        p_label.append(cur_p_label)

    new_img1 = np.concatenate((change_img_arr[0], change_img_arr[1], change_img_arr[2]), axis=1)
    new_img2 = np.concatenate((change_img_arr[3], central_fragment, change_img_arr[4]), axis=1)
    new_img3 = np.concatenate((change_img_arr[5], change_img_arr[6], change_img_arr[7]), axis=1)
    new_img = np.concatenate((new_img1, new_img2, new_img3), axis=0)
    # cv2.imshow('shuffle', new_img)
    # cv2.waitKey(100000)
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, '{0:05d}'.format(cur_id)
                                 + '_' + str(shuffle_list[0]) + '_' + str(shuffle_list[1])
                                 + '_' + str(shuffle_list[2]) + '_' + str(shuffle_list[3])
                                 + '_' + str(shuffle_list[4]) + '_' + str(shuffle_list[5])
                                 + '_' + str(shuffle_list[6]) + '_' + str(shuffle_list[7]) + '.jpg'), new_img)
    return p_label, new_img


def get_reassemble_image(img_arr, bias_list, original_img=None):
    """
    :param img_arr:
    :param bias_list:
    :param original_img:
    :return:
    """
    if original_img is not None:
        out_img = np.array(original_img * 0.2, dtype='uint8')
    else:
        out_img = np.zeros((398, 398, 3), dtype='uint8')
    out_img[bias_list[0]:96 + bias_list[0], bias_list[1]:96 + bias_list[1]] = img_arr[0]
    out_img[bias_list[2]:96 + bias_list[2], 144 + bias_list[3]:240 + bias_list[3]] = img_arr[1]
    out_img[bias_list[4]:96 + bias_list[4], 288 + bias_list[5]:384 + bias_list[5]] = img_arr[2]
    out_img[bias_list[6] + 144:240 + bias_list[6], bias_list[7]:96 + bias_list[7]] = img_arr[3]
    out_img[bias_list[8] + 144:240 + bias_list[8], 144 + bias_list[9]:240 + bias_list[9]] = img_arr[4]
    out_img[bias_list[10] + 144:240 + bias_list[10], 288 + bias_list[11]:384 + bias_list[11]] = img_arr[5]
    out_img[bias_list[12] + 288:384 + bias_list[12], bias_list[13]:96 + bias_list[13]] = img_arr[6]
    out_img[bias_list[14] + 288:384 + bias_list[14], 144 + bias_list[15]:240 + bias_list[15]] = img_arr[7]
    out_img[bias_list[16] + 288:384 + bias_list[16], 288 + bias_list[17]:384 + bias_list[17]] = img_arr[8]
    return out_img


def get_fragment_list(img_arr, cur_id=0, save_unshuffled_path=None):
    """
    :param img_arr: input image array
    :param cur_id: cur_id to set the label and Random Seed
    :param save_unshuffled_path: save the image with dark gap, default is None
    :return: 9 segmentation fragment of image as numpy array; bias list to ensure the image recover

    resize the image into 398*398 shape with gap
    """
    height, width, _ = img_arr.shape
    out_size = (398, 398)

    # resize the image and cut the other
    if width > height:
        cropped_img = img_arr[:, int(width / 2) - math.ceil(height / 2): int(width / 2) + math.floor(height / 2), :]
    elif width < height:
        cropped_img = img_arr[int(height / 2) - math.ceil(width / 2): int(height / 2) + math.floor(width / 2), :, :]
    else:
        cropped_img = img_arr
    resized_img = cv2.resize(cropped_img, out_size)

    # create the fragment
    np.random.seed(WHOLE_IMG_SEED[cur_id])
    bias_list = np.random.randint(15, size=18)
    img1 = resized_img[bias_list[0]:96 + bias_list[0], bias_list[1]:96 + bias_list[1]].copy()
    img2 = resized_img[bias_list[2]:96 + bias_list[2], 144 + bias_list[3]:240 + bias_list[3]].copy()
    img3 = resized_img[bias_list[4]:96 + bias_list[4], 288 + bias_list[5]:384 + bias_list[5]].copy()
    img4 = resized_img[bias_list[6] + 144:240 + bias_list[6], bias_list[7]:96 + bias_list[7]].copy()
    img5 = resized_img[bias_list[8] + 144:240 + bias_list[8], 144 + bias_list[9]:240 + bias_list[9]].copy()
    img6 = resized_img[bias_list[10] + 144:240 + bias_list[10], 288 + bias_list[11]:384 + bias_list[11]].copy()
    img7 = resized_img[bias_list[12] + 288:384 + bias_list[12], bias_list[13]:96 + bias_list[13]].copy()
    img8 = resized_img[bias_list[14] + 288:384 + bias_list[14], 144 + bias_list[15]:240 + bias_list[15]].copy()
    img9 = resized_img[bias_list[16] + 288:384 + bias_list[16], 288 + bias_list[17]:384 + bias_list[17]].copy()

    # set the gap part as darkened place and reassemble the un-darkened fragment as true image
    out_img = get_reassemble_image([img1, img2, img3, img4, img5, img6, img7, img8, img9], bias_list, resized_img)
    if save_unshuffled_path is not None:
        cv2.imwrite(save_unshuffled_path + '/' + '{0:05d}'.format(cur_id) + '.jpg', out_img)
    return [img1, img2, img3, img4, img5, img6, img7, img8, img9, bias_list]


def get_hori(input_path, cur_id=0):
    img = cv2.imread(input_path)
    # create the fragment
    img1, img2, img3, img4, img5, img6, img7, img8, img9 = get_fragment_list(img)

    pos_train_img1 = np.concatenate(([img1], [img2], [img4], [img5], [img7], [img8]))
    pos_train_img2 = np.concatenate(([img2], [img3], [img5], [img6], [img8], [img9]))
    pos_train_img = []
    for i in range(len(pos_train_img1)):
        pos_train_img.append([pos_train_img1[i], pos_train_img2[i]])
    pos_train_img = np.array(pos_train_img)
    pos_train_label = np.array([1] * 6, dtype=bool)

    neg_train_img1 = np.concatenate(([img1], [img1], [img1], [img1], [img1], [img1], [img1],
                                     [img2], [img2], [img2], [img2], [img2], [img2], [img2],
                                     [img3], [img3], [img3], [img3], [img3], [img3], [img3], [img3],
                                     [img4], [img4], [img4], [img4], [img4], [img4], [img4],
                                     [img5], [img5], [img5], [img5], [img5], [img5], [img5],
                                     [img6], [img6], [img6], [img6], [img6], [img6], [img6], [img6],
                                     [img7], [img7], [img7], [img7], [img7], [img7], [img7],
                                     [img8], [img8], [img8], [img8], [img8], [img8], [img8],
                                     [img9], [img9], [img9], [img9], [img9], [img9], [img9], [img9]
                                     ))
    neg_train_img2 = np.concatenate(([img3], [img4], [img5], [img6], [img7], [img8], [img9],
                                     [img1], [img4], [img5], [img6], [img7], [img8], [img9],
                                     [img1], [img2], [img4], [img5], [img6], [img7], [img8], [img9],
                                     [img1], [img2], [img3], [img6], [img7], [img8], [img9],
                                     [img1], [img2], [img3], [img4], [img7], [img8], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img7], [img8], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img7],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img7], [img8]
                                     ))
    random.seed(WHOLE_IMG_SEED[cur_id])
    np.random.shuffle(neg_train_img1)
    random.seed(WHOLE_IMG_SEED[cur_id])
    np.random.shuffle(neg_train_img2)
    neg_train_img = []
    for i in range(len(pos_train_img2)):
        neg_train_img.append([neg_train_img1[i], neg_train_img2[i]])
    neg_train_img = np.array(neg_train_img)

    neg_train_label = np.array([0] * 6, dtype=bool)
    p_img = np.concatenate((pos_train_img, neg_train_img), axis=0)
    p_label = np.concatenate((pos_train_label, neg_train_label), axis=0)
    return p_img, p_label


def get_vert(input_path, cur_id=0):
    img = cv2.imread(input_path)
    # create the fragment
    img1, img2, img3, img4, img5, img6, img7, img8, img9 = get_fragment_list(img)

    pos_train_img1 = np.concatenate(([img1], [img2], [img3], [img4], [img5], [img6]))
    pos_train_img2 = np.concatenate(([img4], [img5], [img6], [img7], [img8], [img9]))
    pos_train_img = []
    for i in range(len(pos_train_img1)):
        pos_train_img.append([pos_train_img1[i], pos_train_img2[i]])
    pos_train_img = np.array(pos_train_img)
    pos_train_label = np.array([1] * 6, dtype=bool)

    neg_train_img1 = np.concatenate(([img1], [img1], [img1], [img1], [img1], [img1], [img1],
                                     [img2], [img2], [img2], [img2], [img2], [img2], [img2],
                                     [img3], [img3], [img3], [img3], [img3], [img3], [img3],
                                     [img4], [img4], [img4], [img4], [img4], [img4], [img4],
                                     [img5], [img5], [img5], [img5], [img5], [img5], [img5],
                                     [img6], [img6], [img6], [img6], [img6], [img6], [img6],
                                     [img7], [img7], [img7], [img7], [img7], [img7], [img7], [img7],
                                     [img8], [img8], [img8], [img8], [img8], [img8], [img8], [img8],
                                     [img9], [img9], [img9], [img9], [img9], [img9], [img9], [img9]
                                     ))
    neg_train_img2 = np.concatenate(([img2], [img3], [img5], [img6], [img7], [img8], [img9],
                                     [img1], [img3], [img4], [img6], [img7], [img8], [img9],
                                     [img1], [img2], [img4], [img5], [img7], [img8], [img8],
                                     [img1], [img2], [img3], [img5], [img6], [img8], [img9],
                                     [img1], [img2], [img3], [img4], [img6], [img7], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img7], [img8],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img8], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img7], [img9],
                                     [img1], [img2], [img3], [img4], [img5], [img6], [img7], [img8]
                                     ))

    random.seed(WHOLE_IMG_SEED[cur_id])
    np.random.shuffle(neg_train_img1)
    random.seed(WHOLE_IMG_SEED[cur_id])
    np.random.shuffle(neg_train_img2)

    neg_train_img = []
    for i in range(len(pos_train_img2)):
        neg_train_img.append([neg_train_img1[i], neg_train_img2[i]])
    neg_train_img = np.array(neg_train_img)

    neg_train_label = np.array([0] * 6, dtype=bool)
    p_img = np.concatenate((pos_train_img, neg_train_img), axis=0)
    p_label = np.concatenate((pos_train_label, neg_train_label))
    return p_img, p_label


def get_8fragment_dataset(input_path, gap_image_path=None, shuffled_fragment_path=None):
    """

    :param input_path: Coco dataset path
    :param gap_image_path: Save gap image path, if do not save image gap_image_path is None
    :param shuffled_fragment_path: Save gap image path, if do not save image gap_image_path is None
    :param bias_list_path: Save bias_list path
    :return: None

    image source from CoCo dataset
    use 10000 image for training and 2000 image for testing
    the central fragment is fixed. the 8 round side fragments is flexible
    save the preprocessed image with gap (the gap part is darkened)
    save the shuffled image after gap process
    save the bias_list to ensure the image recover
    """
    global WHOLE_IMG_NUM, WHOLE_IMG_SEED
    WHOLE_IMG_NUM = 12000
    random.seed(RANDOM_SEED)
    WHOLE_IMG_SEED = [random.randint(0, 1000) for i in range(WHOLE_IMG_NUM)]

    random.seed(RANDOM_SEED)
    image_name_list = os.listdir(input_path)
    random.shuffle(image_name_list)

    if gap_image_path is not None and not os.path.exists(gap_image_path):
        os.mkdir(gap_image_path)
    if shuffled_fragment_path is not None and not os.path.exists(shuffled_fragment_path):
        os.mkdir(shuffled_fragment_path)

    train_bias_list = []
    train_img_arr_list = []
    train_img_label_list = []
    with tqdm(total=10000) as pbar:
        for i in range(10000):
            while True:
                file_name = image_name_list.pop(0)
                img_arr = cv2.imread(os.path.join(input_path, file_name))
                if img_arr is not None:
                    break
            img1, img2, img3, img4, img5, img6, img7, img8, img9, bias_list = get_fragment_list(img_arr,
                                                                                                cur_id=i,
                                                                                                save_unshuffled_path=gap_image_path)
            train_bias_list.append(bias_list)
            img_label, img_arr = get_shuffle_image([img1, img2, img3, img4, img5, img6, img7, img8, img9], cur_id=i,
                                                   save_path=shuffled_fragment_path)
            train_img_arr_list.append(img_arr)
            train_img_label_list.append(img_label)
            pbar.update(1)
    np.save(os.path.join(File_path, 'train_img.npy'), np.array(train_img_arr_list, dtype=np.uint8))
    np.save(os.path.join(File_path, 'train_label.npy'), np.array(train_img_label_list, dtype=bool))
    np.save(os.path.join(File_path, 'train_bias_list.npy'), np.array(train_bias_list, dtype=int))

    test_bias_list = []
    test_img_arr_list = []
    test_img_label_list = []
    with tqdm(total=2000) as pbar:
        for i in range(2000):
            while True:
                file_name = image_name_list.pop(0)
                img_arr = cv2.imread(os.path.join(input_path, file_name))
                if img_arr is not None:
                    break
            img1, img2, img3, img4, img5, img6, img7, img8, img9, bias_list = get_fragment_list(img_arr, cur_id=i+10000, save_unshuffled_path=gap_image_path)
            test_bias_list.append(bias_list)
            img_arr, img_label = get_shuffle_image([img1, img2, img3, img4, img5, img6, img7, img8, img9],
                                                   cur_id=i+10000,
                                                   save_path=shuffled_fragment_path)
            test_img_arr_list.append(img_arr)
            test_img_label_list.append(img_label)
            pbar.update(1)
    np.save(os.path.join(File_path, 'test_img.npy'), np.array(test_img_arr_list, dtype=np.uint8))
    np.save(os.path.join(File_path, 'test_label.npy'), np.array(test_img_label_list, dtype=bool))
    np.save(os.path.join(File_path, 'test_bias_list.npy'), np.array(test_bias_list, dtype=int))


def main():
    coco_path = Coco_path
    get_8fragment_dataset(coco_path,
                          gap_image_path=os.path.join(File_path, "gap_image"),
                          shuffled_fragment_path=os.path.join(File_path, "shuffle_fragment"))


if __name__ == '__main__':
    main()
