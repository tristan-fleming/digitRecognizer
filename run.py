import image_open as io
import simple_preprocess as sp
import feature_extractor as fe
import numpy as np
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def run_feature_extraction(feature_num,num_samples):
    train, valid, test = io.read_MNIST()
    feature_vals = []
    kwargs = dict(histtype = 'stepfilled', alpha = 0.3, normed = True, bins = 40, range = (0,1))
    for digit in range(0,10):
        digit_imgs, digit_indices = io.MNIST_sort(train, digit,num_samples)
        #digit_proc_thres1, digit_proc_thres2 = sp.run_image_preprocess_MNIST(digit_imgs)
        digit_proc = sp.run_image_preprocess_MNIST(digit_imgs)
        #br_digit, holes_digit = fe.features_MNIST(digit_proc_thres1, digit_proc_thres2)
        features = fe.features_MNIST(digit_proc)
        #feature_list = [br, br_s1, br_s2, br_s3, br_s4, br_s5, br_s6,
        #holes, num_lines_img, num_lines_top, num_lines_bottom, num_lines_left,
        #num_lines_right, max_line_length_img, max_line_length_top,
        #max_line_length_bottom, max_line_length_left, max_line_length_right]
        feature_vals.append(list(x[feature_num] for x in features))
    plt.hist(feature_vals, **kwargs, label = "%d" %(digit), cmap=plt.cm.RdYlGn)
    plt.tight_layout()
    plt.legend(loc = 'upper right')
    plt.show()
    feature_vals = np.array(feature_vals)
    return feature_vals

def run_classification(num_train, num_valid):
    train, valid, test = io.read_MNIST()
    # Use train_test_split to pick a random sampling of the training and validation sets
    train1, train2, train1_labels, train2_labels = train_test_split(train[0], train[1], train_size = num_train)
    digit_imgs_train_np = [np.array(x).reshape(28,28) for x in train1]
    ytrain = train1_labels
    digit_train_proc = sp.run_image_preprocess_MNIST(digit_imgs_train_np)
    Xtrain = fe.features_MNIST(digit_train_proc)
    valid1, valid2, valid1_labels, valid2_labels = train_test_split(valid[0], valid[1], train_size = num_valid)
    digit_imgs_valid_np = [np.array(x).reshape(28,28) for x in valid1]
    yvalid = valid1_labels
    digit_valid_proc = sp.run_image_preprocess_MNIST(digit_imgs_valid_np)
    Xvalid = fe.features_MNIST(digit_valid_proc)
    #min_max_scaler = MinMaxScaler()
    Xvalid_scale = scale(Xvalid, axis = 0)
    Xtrain_scale = scale(Xtrain, axis = 0)
    # Test supervised learning
    percent_corr = []
    estimators = []
    models = [
        GaussianNB(),
        KNeighborsClassifier(3, weights = 'uniform'),
        KNeighborsClassifier(3, weights = 'distance'),
        KNeighborsClassifier(5, weights = 'uniform'),
        KNeighborsClassifier(5, weights = 'distance'),
        KNeighborsClassifier(7, weights = 'uniform'),
        KNeighborsClassifier(7, weights = 'distance'),
        KNeighborsClassifier(9, weights = 'uniform'),
        KNeighborsClassifier(9, weights = 'distance'),
        KNeighborsClassifier(11, weights = 'uniform'),
        KNeighborsClassifier(11, weights = 'distance'),
        DecisionTreeClassifier(max_depth = 3),
        DecisionTreeClassifier(max_depth = 5),
        DecisionTreeClassifier(max_depth = 7),
        RandomForestClassifier(max_depth = 5, n_estimators = 10),
        SVC(kernel = "linear"),
        SVC(kernel = "linear", C = 0.25),
        SVC(kernel = "linear", C = 0.025),
        LinearSVC(multi_class = 'ovr'),
        LinearSVC(multi_class = 'ovr', C = 0.25),
        LinearSVC(multi_class = 'ovr', C = 0.025),
        LinearSVC(multi_class = 'crammer_singer'),
        AdaBoostClassifier(),
        MLPClassifier(alpha=1),
        QuadraticDiscriminantAnalysis()]
    model_strings = ['gnb','3nn_uniform','3nn_dist','5nn_uniform',
        '5nn_dist','7nn_uniform','7nn_dist','9nn_uniform',
        '9nn_dist','11nn_uniform','11nn_dist','dt3', 'dt5', 'dt7',
        'dt5ne10','rf','svc_lin','svc_lin_c025','svc_lin_c0025',
        'lin_svc_ovr','lin_svc_ovr_c025','lin_svc_ovr_c0025',
        'lin_svc_cramm','ada_boost','qda']
    for i,model in enumerate(models):
        model.fit(Xtrain_scale, ytrain)
        y_model = model.predict(Xvalid_scale)
        percent_corr.append(accuracy_score(yvalid, y_model))
        estimators.append((model_strings[i], model))
        eclf = VotingClassifier(estimators = estimators, voting = 'hard')

    eclf.fit(Xtrain_scale, ytrain)
    y_model = eclf.predict(Xvalid_scale)
    percent_corr.append(accuracy_score(yvalid, y_model))
    return percent_corr
