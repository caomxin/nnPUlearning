import os
import pickle
import tarfile
import urllib.request

import numpy as np


def split_dataset(x_dataset, ratio):
    # x_arr = np.arange(x_dataset.size)
    # np.random.shuffle(x_arr)

    x_num_train = int(ratio * (x_dataset.shape)[0])
    x_train = x_dataset[0:x_num_train]
    x_test = x_dataset[x_num_train:(x_dataset.shape)[0]]
    return x_train, x_test


def get_mnist():
    # mnist = fetch_mldata('MNIST original', data_home=".")
    # Alternative method to load MNIST, if mldata.org is down
    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")
    x = mnist['data']
    y = mnist['target']
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)


def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = -1  # 5074 elements set to -1
    # list(testY).count(-1)
    return trainY, testY


def get_drugbank():
    print('********** date preparing ************')

    npzfile = np.load("a.npz")

    # drug <class 'numpy.ndarray'>   (6386, 1000)
    drug_data = npzfile['drug_data']
    print(type(drug_data), " ", drug_data.shape)

    drug_data_train, drug_data_test = split_dataset(drug_data, 0.7)

    # target <class 'numpy.ndarray'>   (4154, 1500)
    target_data = npzfile['target_data']
    print(type(target_data), " ", target_data.shape)

    # known_relationship <class 'numpy.ndarray'>   (6386, 4154)
    knownRelationship = npzfile['knownRelationship']
    print(type(knownRelationship), " ", knownRelationship.shape)

    # known_drug_target_pair <class 'numpy.ndarray'>   (15360, 2)
    known_drugtarget = npzfile['known_drugdtarget']
    print(type(known_drugtarget), " ", known_drugtarget.shape)

    list_of_pairs = known_drugtarget.tolist()  # a list of known pairs, length = 15360

    ### test with 100,000 pairs in which 15360 are positive labeled
    # drug_target_combined is of shape([15360,2])
    drug_target_combined = []

    for i in range(known_drugtarget.shape[0]):
        temp_pair = []
        temp_pair.append(np.array(drug_data[known_drugtarget[i][0]]))
        temp_pair.append(np.array(target_data[known_drugtarget[i][1]]))
        temp_pair = np.array(temp_pair)
        drug_target_combined.append(temp_pair)

        # Since we are adding all known pairs (labeled pairs), the crspd y value should be 1
        # alternatively, we can directly initialize y as y = np.ones((15360,))
        # y.extend([1])

    drug_target_combined = np.array(drug_target_combined)
    assert (drug_target_combined.shape == (15360, 2))
    y = np.ones(drug_target_combined.shape[0])
    print("*** Known drug-target pair preparation done ***")

    ## add another 100,000 - 15360 unlabeled samples to the dataset
    drug_target_combined = drug_target_combined.tolist()
    y = y.tolist()
    # i is used for counting
    i = 0
    while (i < 100000 - 15360):
        j = np.random.randint(4154)
        k = np.random.randint(6386)
        if [k, j] in list_of_pairs:
            continue

        temp_pair = []
        temp_pair.append(np.array(drug_data[k]))
        temp_pair.append(np.array(target_data[j]))
        temp_pair = np.array(temp_pair)
        drug_target_combined.append(temp_pair)
        y.extend([-1])
        i = i + 1

    y = np.array(y)
    drug_target_combined = np.array(drug_target_combined)
    assert (y.shape[0] == 100000)
    assert (drug_target_combined.shape == (100000, 2))

    # these two parts can be combined together later, but now let's just make it work

    # Shuffle around the drug_target_combined
    perm = np.random.permutation(len(drug_target_combined))  # should be of length 100000
    drug_target_combined, y = drug_target_combined[perm], y[perm]

    x_tr = np.asarray(drug_target_combined[:70000], dtype=np.float32)
    y_tr = np.asarray(y[:70000], dtype=np.int32)
    x_te = np.asarray(drug_target_combined[70000:], dtype=np.float32)
    y_te = np.asarray(y[70000:], dtype=np.int32)

    return (x_tr, y_tr), (x_te, y_te)


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(path="./mldata"):
    if not os.path.isdir(path):
        os.mkdir(path)
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.basename(url)
    full_path = os.path.join(path, file_name)
    folder = os.path.join(path, "cifar-10-batches-py")
    # if cifar-10-batches-py folder doesn't exists, download from website
    if not os.path.isdir(folder):
        print("download the dataset from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, full_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0, 32 * 32 * 3))
    y_tr = np.empty(1)
    for i in range(1, 6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            x_tr = data_dict['data']
            y_tr = data_dict['labels']
        else:
            x_tr = np.vstack((x_tr, data_dict['data']))
            y_tr = np.hstack((y_tr, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    x_te = data_dict['data']
    y_te = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    # label_names = bm['label_names']
    # rehape to (#data, #channel, width, height)
    x_tr = np.reshape(x_tr, (np.shape(x_tr)[0], 3, 32, 32)).astype(np.float32)
    x_te = np.reshape(x_te, (np.shape(x_te)[0], 3, 32, 32)).astype(np.float32)
    # normalize
    x_tr /= 255.
    x_te /= 255.
    return (x_tr, y_tr), (x_te, y_te)  # , label_names


def binarize_cifar10_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY == 2) | (_trainY == 3) | (_trainY == 4) | (_trainY == 5) | (_trainY == 6) | (_trainY == 7)] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY == 2) | (_testY == 3) | (_testY == 4) | (_testY == 5) | (_testY == 6) | (_testY == 7)] = -1
    return trainY, testY


def make_dataset(dataset, n_labeled, n_unlabeled):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert (len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_lp = labeled
        n_n = (Y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        print(X.shape)
        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)  # (10000,1,28,28)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)  # (10000,)
        perm = np.random.permutation(len(Y))  # (10000,)
        X, Y = X[perm], Y[perm]  # shuffle
        return X, Y

    # dataset consists of (trainX, trainY) and (testX, testY) the following line break them up from the argument
    (_trainX, _trainY), (_testX, _testY) = dataset

    # trainX is of shape (60000, 1, 28, 28);
    # trainY is of shape (60000,)
    # testX is of shape (10000, 1, 28, 28)
    # tesetY is of shape(10000,)
    trainX, trainY, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY)
    testX, testY = make_PN_dataset_from_binary_dataset(_testX, _testY)
    print("training:{}".format(trainX.shape))
    print("test:{}".format(testX.shape))
    return list(zip(trainX, trainY)), list(zip(testX, testY)), prior


def load_dataset(dataset_name, n_labeled, n_unlabeled):
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)
    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar10_class(trainY, testY)
    elif dataset_name == 'drugbank':
        (trainX, trainY), (testX, testY) = get_drugbank()
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    XYtrain, XYtest, prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled)
    return XYtrain, XYtest, prior
