from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
def imagee():
    image = Image.open("Lion.jpeg")
    print(image.format, image.size, image.mode)
    # image.show()
    r, g, b = image.split()
    r_pix, g_pix, b_pix = np.array(r), np.array(g), np.array(b),

    U_r_svd, S_r_svd, VT_r_svd = np.linalg.svd(r_pix, full_matrices=True, compute_uv=True)
    U_b_svd, S_b_svd, VT_b_svd = np.linalg.svd(b_pix, full_matrices=True, compute_uv=True)
    U_g_svd, S_g_svd, VT_g_svd = np.linalg.svd(g_pix, full_matrices=True, compute_uv=True)

    k = 100

    Ak_r = np.dot(U_r_svd[:, :k], np.dot(np.diag(S_r_svd[:k]), VT_r_svd[:k, :]))
    Ak_g = np.dot(U_g_svd[:, :k], np.dot(np.diag(S_g_svd[:k]), VT_g_svd[:k, :]))
    Ak_b = np.dot(U_b_svd[:, :k], np.dot(np.diag(S_b_svd[:k]), VT_b_svd[:k, :]))

    imageR = Image.fromarray(Ak_r.astype('uint8'))
    imageG = Image.fromarray(Ak_g.astype('uint8'))
    imageB = Image.fromarray(Ak_b.astype('uint8'))

    rgb_image = Image.merge("RGB", (imageR, imageG, imageB))
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
    U ,_ ,_ = np.linalg.svd(center_data, full_matrices=False)
    Us=U[:,:s]
    train_proj = np.matmul(Us.T, data)
    test_proj = np.matmul(Us.T, test_data)
    return train_proj,test_proj


def open_test_data():
    my_dict = unpickle("cifar-10-batches-py/test_batch")
    all_img = my_dict[b'data']
    all_img1 = gray_data(all_img)
    return all_img1.T


def build_distance_matrix(train_data, test_data):
    num_test = test_data.shape[1]
    num_train = train_data.shape[1]
    dis_matrix = np.zeros(num_test,num_train)
    for img_test in range(num_test):
        for img_train in range(num_train):
            dis_matrix[img_test][img_train] = np.linalg.norm(test_data[:, img_test] - train_data[:, img_train])
    return dis_matrix


def Knn(train_data, train_labels, test_data, k):
    dist_matrix = build_distance_matrix(train_data, test_data)
    num_test = test_data.shape[1]
    num_train = train_data.shape[1]
    y_pred = np.zeros(num_test)

    for i in range(num_test):
        # Find k-nearest neighbors for each test sample
        nearest_indices = np.argsort(dist_matrix[i])[:k]
        nearest_labels = train_labels[nearest_indices]

        # Predict the label based on majority vote
        y_pred[i] = np.argmax(np.bincount(nearest_labels))

    return y_pred




if __name__ == '__main__':
    # this is an example. Your path to file may be different
    data ,labels_vec = marge_data()
    center_data = centering_data(data)
    test_data = open_test_data()
    center_test_data = centering_data(data)
    pca_data ,pca_test = pca(data,test_data,5)



