from math import radians, cos, sin, asin, sqrt, floor, ceil
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import time


def random_color():
    names = {}
    idx = 0
    for name, hex in matplotlib.colors.cnames.items():
        names[idx] = name
        idx += 1
    return names[random.randint(0, idx)]


def get_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis


def get_grid_id(lon, lat):
    x = floor(get_distance(minLatitude, minLongitude, minLatitude, lon) / grid_stride)
    y = floor(get_distance(minLatitude, maxLongitude, lat, maxLongitude) / grid_stride)
    return x + y * colGridSize


def get_gongcan_map(gongcanDf):
    tmpMap = {}
    for i in range(len(gongcanDf)):
        key = gongcanDf.iloc[i]['RNCID'] + '_' + gongcanDf.iloc[i]['CellID']
        value = [gongcanDf.iloc[i]['Longitude'], gongcanDf.iloc[i]['Latitude']]
        tmpMap[key] = value
    return tmpMap


def process_data():
    data = np.zeros((size, 49))
    labels = np.zeros((size, 3))
    for index in range(len(dataDf)):
        num_connected = dataDf.iloc[index]['Num_connected']
        item = dataDf.iloc[index]
        lon = item['Longitude']
        lat = item['Latitude']
        gridId = get_grid_id(lon, lat)
        labels[index][0] = gridId
        labels[index][1] = lon
        labels[index][2] = lat
        time = item['MRTime']
        minMR = [0] * 7
        for j in range(num_connected):
            key = str(item[features[2] + str(j + 1)]) + '_' + str(item[features[3] + str(j + 1)])
            location = gongcanMap.get(key)
            if location is not None:
                data[index][7 * j + 0] = time
                data[index][7 * j + 1] = location[0]
                data[index][7 * j + 2] = location[1]
                for k in range(3, 7):
                    data[index][7 * j + k] = item[features[k - 3] + str(j + 1)]
                # get min
                if data[index][7 * j + 3] < minMR[3]:
                    minMR = data[index][7 * j: 7 * (j + 1)]
        for j in range(num_connected, 7):
            data[index][7 * (num_connected - 1): 7 * num_connected] = minMR
    return data, labels


def generatebatch(X, Y, n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys


def init_grip_number(y_test):
    grid_number_map = {}
    # initialize the grid
    for i in range(int(get_grid_id(maxLongitude, maxLatitude))):
        grid_number_map[i] = 0

    for i in range(len(y_test)):
        grid_number_map[int(y_test[i][0])] = grid_number_map[int(y_test[i][0])] + 1
    return grid_number_map


def get_max_grids(grid_number_map):
    grid_number_map = sorted(grid_number_map.items(), key=lambda d: d[1], reverse=True)
    max_grids = {}
    for i in range(10):
        max_grids[grid_number_map[i][0]] = grid_number_map[i][1]
    return max_grids


def evaluate(y_pred, y_test, max_grids):
    keys = list(max_grids.keys())
    dt = 0
    d = 0
    t = 0
    for i in range(len(keys)):
        t = t + max_grids[keys[i]]
        for j in range(len(y_pred)):
            if int(keys[i]) == y_pred[j] == y_test[j]:
                dt += 1
            if int(keys[i]) == y_pred[j]:
                d += 1
    if d == 0:
        precision = 0
    else:
        precision = dt / d
    if t == 0:
        recall = 0
    else:
        recall = dt / t
    if precision + recall == 0:
        f_measurement = 0
    else:
        f_measurement = 2 * precision * recall / (precision + recall)
    return [round(precision, 4), round(recall, 4), round(f_measurement, 4),
            round(sum(y_pred == y_test) / len(y_test), 4)]


def make_precise_picture(precises, names):
    plt.figure(figsize=(15, 8))
    length = precises.shape[0]
    x1 = [i - 0.3 for i in range(1, length + 1)]
    x2 = [i - 0.1 for i in range(1, length + 1)]
    x3 = [i + 0.1 for i in range(1, length + 1)]
    x4 = [i + 0.3 for i in range(1, length + 1)]
    l1 = plt.bar(x=x1, height=precises[:, 0].tolist(), width=0.2, color=random_color(), label='precise')
    l2 = plt.bar(x=x2, height=precises[:, 1].tolist(), width=0.2, color=random_color(), label='recall')
    l3 = plt.bar(x=x3, height=precises[:, 2].tolist(), width=0.2, color=random_color(), label='f-measurement')
    l4 = plt.bar(x=x4, height=precises[:, 3].tolist(), width=0.2, color=random_color(), label='accurate')
    plt.xlabel('convolution way')
    plt.ylabel('score (%)')
    plt.title('Result comparison histogram')
    plt.xticks(x2, names)
    for index in range(length):
        plt.text(x1[index], precises[index, 0], '%.2f' % precises[index, 0], ha='center', va='bottom')
        plt.text(x2[index], precises[index, 1], '%.2f' % precises[index, 1], ha='center', va='bottom')
        plt.text(x3[index], precises[index, 2], '%.2f' % precises[index, 2], ha='center', va='bottom')
        plt.text(x4[index], precises[index, 3], '%.2f' % precises[index, 3], ha='center', va='bottom')
    plt.legend(handles=[l1, l2, l3, l4], labels=['precise', 'recall', 'f-measurement', 'accurate'], loc='best')
    plt.show()


def make_time_picture(times, names):
    plt.figure(figsize=(10, 5))
    l1 = plt.bar(x=range(1, len(times) + 1), height=times, width=0.3, color=random_color())
    plt.xlabel('classifier')
    plt.ylabel('cost(s)')
    plt.title('Time performance bar diagram')
    length = len(times)
    idx = [i - 0.3 for i in range(1, length + 1)]
    plt.xticks(idx, names)
    for index in range(len(times)):
        plt.text(idx[index], times[index], '%.2f' % times[index], ha='center', va='bottom')
    plt.legend(handles=[l1], labels=['cost'], loc='best')
    plt.show()


def convolution(convolution_size):
    start = time.time()
    in_channels = 1
    channel_multiplier = initial_out_channels
    # convolution layer 1 + activate layer 1
    conv_filter_w1 = tf.Variable(tf.random_normal(convolution_size + [in_channels, channel_multiplier]))
    conv_filter_b1 = tf.Variable(tf.random_normal([channel_multiplier]))
    conv_out1 = tf.nn.conv2d(tf_X, conv_filter_w1, strides=[1, 1, 1, 1], padding='SAME')

    relu_feature_maps1 = tf.nn.relu6(conv_out1 + conv_filter_b1)

    conv_size = [relu_feature_maps1.shape[1].value, relu_feature_maps1.shape[2].value]

    in_channels_2 = channel_multiplier
    channel_multiplier_2 = initial_out_channels
    # convolution layer 2
    conv_filter_w2 = tf.Variable(tf.random_normal(conv_size + [in_channels_2, channel_multiplier_2]))
    conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2, strides=[1, 1, 1, 1], padding='SAME')
    b2_size = in_channels * channel_multiplier_2
    conv_filter_b2 = tf.Variable(tf.random_normal([b2_size]))
    conv_out2 = conv_out2 + conv_filter_b2
    # BN + activate layer 1
    batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([b2_size]))
    scale = tf.Variable(tf.ones([b2_size]))
    epsilon = 1e-3
    BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)

    relu_BN_maps2 = tf.nn.relu6(BN_out)

    relu_BN_maps2_size = relu_BN_maps2.shape[1].value * relu_BN_maps2.shape[2].value

    max_pool2_flat = tf.reshape(relu_BN_maps2, [-1, relu_BN_maps2_size * b2_size])

    # all connection layer
    fc_w1 = tf.Variable(tf.random_normal([relu_BN_maps2_size * b2_size, sample_size]))
    fc_b1 = tf.Variable(tf.random_normal([sample_size]))
    fc_out1 = tf.nn.relu6(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)

    # 输出层
    out_w1 = tf.Variable(tf.random_normal([sample_size, sample_size]))
    out_b1 = tf.Variable(tf.random_normal([sample_size]))
    pred = tf.nn.softmax(tf.matmul(fc_out1, out_w1) + out_b1)

    loss = -tf.reduce_mean(tf_Y * tf.log(tf.clip_by_value(pred, 1e-11, 1.0)))

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    y_pred = tf.argmax(pred, 1)

    res = []
    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_xs, batch_ys in generatebatch(X, Y, Y.shape[0], batch_size):
                sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})
            if epoch % 10 == 0:
                y = sess.run(y_pred, feed_dict={tf_X: X, tf_Y: Y})
                res = evaluate(y, y_test, max_grids)
                print(epoch, res)
    end = time.time()
    return res, (end - start)


def draw_from_csv():
    df = pd.read_csv('../output.csv', header=0)
    values = df.values
    precises = np.zeros((values.shape[0], values.shape[1] - 1))
    conv_names = []
    for i in range(len(values)):
        precises[i] = np.array(values[i][1:])
        conv_names.append(values[i][0])
    # draw comparable picture
    make_precise_picture(precises, conv_names)


if __name__ == '__main__':
    grid_stride = 150
    # read data
    dataDf = pd.read_csv('../data_2g.csv', header=0, dtype={'RNCID_1': np.object, 'CellID_1': np.object,
                                                         'RNCID_2': np.object, 'CellID_2': np.object,
                                                         'RNCID_3': np.object, 'CellID_3': np.object,
                                                         'RNCID_4': np.object, 'CellID_4': np.object,
                                                         'RNCID_5': np.object, 'CellID_5': np.object,
                                                         'RNCID_6': np.object, 'CellID_6': np.object,
                                                         'RNCID_7': np.object, 'CellID_7': np.object})

    gongcanDf = pd.read_csv('../2g_gongcan.csv', header=0, dtype={'RNCID': np.object, 'CellID': np.object})
    # get gongcan data
    gongcanMap = get_gongcan_map(gongcanDf)

    size = len(dataDf)
    # Get the matrix
    minLatitude = dataDf['Latitude'].min()
    maxLatitude = dataDf['Latitude'].max()
    minLongitude = dataDf['Longitude'].min()
    maxLongitude = dataDf['Longitude'].max()

    # compute the grid row and col number
    colGridSize = ceil(get_distance(minLatitude, minLongitude, minLatitude, maxLongitude) / grid_stride)
    rowGridSize = ceil(get_distance(minLatitude, minLongitude, maxLatitude, minLongitude) / grid_stride)

    # feature
    features = ['RSSI_', 'AsuLevel_', 'RNCID_', 'CellID_']
    #
    data, labels = process_data()

    # get grid number map
    # grid_number_map = init_grip_number(labels)
    # max_grids = get_max_grids(grid_number_map)

    scaler = MinMaxScaler()
    # X_data = data
    X_data = scaler.fit_transform(data)
    Y_tmp = labels[:, 0].reshape(-1, 1)

    # one hot code
    Y = OneHotEncoder().fit_transform(Y_tmp).todense()

    y_test = []
    for i in range(Y.shape[0]):
        y_test.append(np.argmax(Y[i, :]))
    y_test = np.array(y_test)
    grid = {}
    for i in range(Y.shape[1]):
        grid[i] = Y[:, i].sum()
    # get the ten max grid meta data
    max_grids = get_max_grids(grid)
    # reshape the data
    X = X_data.reshape(-1, 7, 7, 1)
    # sample size
    sample_size = Y.shape[1]

    initial_out_channels = sample_size // 8

    tf.reset_default_graph()
    # input layer
    tf_X = tf.placeholder(tf.float32, [None, 7, 7, 1])
    tf_Y = tf.placeholder(tf.float32, [None, sample_size])
    # use MGDB algorithm, set the batch size is 1000
    batch_size = 1000
    # convolution size
    convolution_sizes = [[2, 2], [3, 3], [4, 4]]
    # the number of iterators
    epochs = 1000
    # result
    res = []
    costs = []
    # title
    conv_names = ['convolution 2 * 2', 'convolution 3 * 3', 'convolution 4 * 4']
    # train
    for convolution_size in convolution_sizes:
        result, cost = convolution(convolution_size)
        res.append(result)
        costs.append(cost)
    precises = np.array(res)
    # draw comparable picture
    make_precise_picture(precises, conv_names)
    # draw time picture
    make_time_picture(costs, conv_names)
