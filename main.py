from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
def Q4():
    image = Image.open("Lion.jpeg")
    print(image.format, image.size, image.mode)
    #image.show()
    r, g, b = image.split()
    r_pix, g_pix, b_pix = np.array(r), np.array(g), np.array(b),

    U_r_svd, S_r_svd, VT_r_svd = np.linalg.svd(r_pix, full_matrices=True, compute_uv=True)
    U_b_svd, S_b_svd, VT_b_svd = np.linalg.svd(b_pix, full_matrices=True, compute_uv=True)
    U_g_svd, S_g_svd, VT_g_svd = np.linalg.svd(g_pix, full_matrices=True, compute_uv=True)

    k = 150

    Ak_r = np.dot(U_r_svd[:, :k], np.dot(np.diag(S_r_svd[:k]), VT_r_svd[:k, :]))
    Ak_g = np.dot(U_g_svd[:, :k], np.dot(np.diag(S_g_svd[:k]), VT_g_svd[:k, :]))
    Ak_b = np.dot(U_b_svd[:, :k], np.dot(np.diag(S_b_svd[:k]), VT_b_svd[:k, :]))

    imageR = Image.fromarray(Ak_r.astype('uint8'))
    imageG = Image.fromarray(Ak_g.astype('uint8'))
    imageB = Image.fromarray(Ak_b.astype('uint8'))

    rgb_image = Image.merge("RGB", (imageR, imageG, imageB))
    print(rgb_image)
    rgb_image.show()
    print(rgb_image.format, rgb_image.size, rgb_image.mode)

    error = np.sum(np.square((S_r_svd)[k:])) / np.sum(np.square((S_r_svd)))
    print(error)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def marge_data():
    my_dict = unpickle("cifar-10-batches-py/data_batch_1")
    all_img = my_dict[b'data']
    labels_vec= my_dict[b'labels']
    for i in range(2,6):
        file_path = "cifar-10-batches-py/data_batch_"+str(i)
        temp_dict = unpickle(file_path)
        temp_img = temp_dict[b'data']
        temp_labels_vec = temp_dict[b'labels']
        all_img = np.concatenate((all_img, temp_img), axis=0)
        labels_vec += temp_labels_vec
    all_img1= gray_data(all_img)
    return all_img1.T,labels_vec


def gray_data(data):
    data_list = []
    for single_img in data:
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        image = image.convert("L")
        my_array = np.array(image).flatten()
        data_list.append(my_array)
    X = np.asarray(data_list)
    return X



def centering_data(data):
    mean = np.mean(data, axis=1)#calculates the mean along each sample (each row) of the dataset.
    center_data = np.apply_along_axis(lambda x: x - mean, 0, data)
    return center_data

def pca(data,test_data,s):
    U ,_ ,_ = np.linalg.svd(data, full_matrices=False)
    Us=U[:,:s]
    train_proj = np.matmul(Us.T, data)
    test_proj = np.matmul(Us.T, test_data)
    return train_proj,test_proj


def open_test_data():
    my_dict = unpickle("cifar-10-batches-py/test_batch")
    all_img = my_dict[b'data']
    labels_vec= my_dict[b'labels']
    all_img1 = gray_data(all_img)
    return all_img1.T,labels_vec





def build_distance_matrix(train_data, test_data):
   num_test = test_data.shape[1]
   num_train = train_data.shape[1]
   distances = np.empty((num_test, num_train))
   for img_test in range(num_test):
       distances[img_test] = np.linalg.norm(test_data[:, img_test, None] - train_data, axis=0)
   return distances.T




def knn(test, train, labels, k):
   distance = build_distance_matrix(test, train)
   test_labels = np.zeros(distance.shape[0])
   for img in range(distance.shape[0]):
       nearest_labels = []
       nearest_k = np.argsort(distance[img])[:k]
       for i in range(k):
           nearest_labels.append(labels[nearest_k[i]])
       test_labels[img] = max(set(nearest_labels), key=nearest_labels.count)
   return test_labels




def comput_error_rate(y_predicted, y_true):
 error_rate = np.mean(y_true != y_predicted.astype(int))
 return error_rate




if __name__ == '__main__':
    #Q4()

    data ,labels_vec = marge_data()
    center_data = centering_data(data)
    test_data,true_labels = open_test_data()
    center_test_data = centering_data(data)
    k_list = [5,10,50,100,500]
    s_list = [1, 10, 500,1024]
    for s in s_list:
        pca_data ,pca_test = pca(data,test_data,s)
        for k in k_list:
           test_prediction = knn(pca_test,pca_data, labels_vec, k)
           error_rate = comput_error_rate(test_prediction, true_labels)
           print("for k = ", k, ", s = ", s, "the error rate is: ", error_rate)






