#Import functionality from other modules

import image_open as io
import simple_preprocess as sp
import feature_extractor as fe

#Import functionality from python packages

import numpy as np
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from matplotlib import cm
from yellowbrick.features.rankd import Rank1D, Rank2D
from mpl_toolkits.mplot3d import Axes3D

def run_feature_extraction(feature_num, num_samples, scaled = False):
    '''Extracts and plots the feature vals for feature_num of the feature vector
    for num_samples of the training MNIST data set. The feature values are
    scaled if the boolean scaled is set to True.'''
    train, valid, test = io.read_MNIST()
    feature_vals = []
    kwargs = dict(alpha = 1, normed = False, bins = 60) #range = (-3,3))
    colors = [cm.tab10(x) for x in np.linspace(0,1,10)]
    X_plot = np.linspace(0, 1, 1000)
    feature_names = ['Blackness ratio', 'Blackness ratio upper left',
    'Blackness ratio upper right', 'Blackness ratio center left',
    'Blackness ratio center right', 'Blackness ratio lower left',
    'Blackness ratio lower right', 'Number of holes',
    'Whiteness ratio without holes',
    'Hough transform: total number of lines',
    'Hough transform: longest line length',
    'Hough transform: longest line in upper half',
    'Hough transform: longest line in lower half',
    'Hough transform: longest line in left half',
    'Hough transform: longest line in right half',
    'Number of lines with angles in quadrants 1/3',
    'Number of lines with angles in quadrants 2/4',
    'Number of vertical lines',
    'Number of horizontal lines']
    for ind in range(0, 36):
        feature_names.append('Histogram of oriented gradients: orientation/block # %d' %(ind))

    plt.close('all')
    plt.figure(1)
    fig, ax = plt.subplots()
    for digit in range(0,10):
        digit_imgs, digit_indices = io.MNIST_sort(train, digit, num_samples)
        #digit_proc_thres1, digit_proc_thres2 = sp.run_image_preprocess_MNIST(digit_imgs)
        digit_proc = sp.run_image_preprocess_MNIST(digit_imgs)
        #br_digit, holes_digit = fe.features_MNIST(digit_proc_thres1, digit_proc_thres2)
        features = fe.features_MNIST(digit_proc)
        if scaled == True:
            features_scale = scale(features, axis  = 0)
            features_scale = features_scale.tolist()
            feature_mx = features_scale
        elif scaled == False:
            feature_mx = features.tolist()
        feature_vals_digit = list(x[feature_num] for x in feature_mx)
        feature_vals.append(feature_vals_digit)
        X = np.asarray(feature_vals_digit)
        #kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.05).fit(X[:,np.newaxis])
        #log_dens = kde.score_samples(X_plot[:,np.newaxis])
        #plt.plot(X_plot, np.exp(log_dens), label = "%d" %(digit), color = colors[digit])
        plt.hist(feature_vals_digit, **kwargs, label = "%d" %(digit), color = colors[digit])
        #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        #plt.plot(bincenters, y, '-', color = colors[digit])
    plt.tight_layout()
    plt.title(feature_names[feature_num], fontsize = 16, fontweight = 'bold')
    #plt.subplots_adjust(top=0.85, wspace = 0.25, hspace = 0.25)
    plt.ylabel('Num. of occurences in %d samples' %(num_samples), fontsize = 12)
    plt.subplots_adjust(top=0.85, left = 0.1)
    plt.savefig("feature_num_%d_all.png" % (feature_num))
    plt.close()

    plt.figure(2)
    fig, ax = plt.subplots(2, 5, figsize=(35,15)) #sharey=True, sharex = True)
    for i, axi in enumerate(ax.flat):
        axi.hist(feature_vals[i], **kwargs, label = "%d" %(digit), color = colors[i])
        axi.tick_params(labelsize = 28)
        axi.set_title('"%d"' %(i), fontsize = 32)
        #axi.set_xlabel(feature_names[feature_num], fontsize = 20)
        #axi.set_xlabel('Scaled blackness ratio [unitless]', fontsize = 12)
    #plt.tight_layout(wspace = 0.5, hspace = 0.5)
    plt.tight_layout()
    fig.text(0.02, 0.5, 'Num. of occurences in %d samples' %(num_samples), va='center', rotation='vertical', fontsize = 36)
    #plt.legend(loc = 'best', frameon = True, fontsize = 12)
    plt.suptitle(feature_names[feature_num], fontsize = 48, fontweight = 'bold')
    plt.subplots_adjust(top=0.9, wspace = 0.15, hspace = 0.25, left = 0.075)
    plt.savefig("feature_num_%d_ind.png" % (feature_num))
    plt.close()
    #feature_vals = np.asarray(feature_vals)
    return

def run_classification(num_train, num_valid, test_set = 'valid'):
    train, valid, test = io.read_MNIST()
    # Use train_test_split to pick a random sampling of the training and validation sets
    train1, train2, train1_labels, train2_labels = train_test_split(train[0], train[1], train_size = num_train)
    digit_imgs_train_np = [np.array(x).reshape(28,28) for x in train1]
    ytrain = train1_labels
    digit_train_proc = sp.run_image_preprocess_MNIST(digit_imgs_train_np)
    Xtrain = fe.features_MNIST(digit_train_proc)
    if test_set == 'valid':
        valid1, valid2, valid1_labels, valid2_labels = train_test_split(valid[0], valid[1], train_size = num_valid)
    elif test_set == 'test':
        valid1, valid2, valid1_labels, valid2_labels = train_test_split(test[0], test[1], train_size = num_valid)
    else:
        print("Valid test_set strings are 'valid' for classifier validation and 'test' for final classifier testing.")
        return
    digit_imgs_valid_np = [np.array(x).reshape(28,28) for x in valid1]
    yvalid = valid1_labels
    digit_valid_proc = sp.run_image_preprocess_MNIST(digit_imgs_valid_np)
    Xvalid = fe.features_MNIST(digit_valid_proc)
    #min_max_scaler = MinMaxScaler()
    Xvalid_scale = scale(Xvalid, axis = 0)
    Xtrain_scale = scale(Xtrain, axis = 0)
    # Test supervised learning
    percent_corr = []
    confusion_matrices = []
    precisions = []
    recalls = []
    fscores = []
    supports = []
    #precision = []
    #recall = []
    #f1 = []
    estimators = []
    models = [
        #GaussianNB(),
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
        #DecisionTreeClassifier(max_depth = 3),
        #DecisionTreeClassifier(max_depth = 5),
        DecisionTreeClassifier(max_depth = 7),
        RandomForestClassifier(max_depth = 5, n_estimators = 10),
        SVC(kernel = "linear"),
        SVC(kernel = "linear", C = 0.25),
        SVC(kernel = "linear", C = 0.025),
        LinearSVC(multi_class = 'ovr'),
        LinearSVC(multi_class = 'ovr', C = 0.25),
        LinearSVC(multi_class = 'ovr', C = 0.025),
        LinearSVC(multi_class = 'crammer_singer'),
        #AdaBoostClassifier(),
        MLPClassifier(alpha=1),
        #QuadraticDiscriminantAnalysis()
        ]
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
        confusion_matrices.append(confusion_matrix(yvalid, y_model))
        precision_classifier, recall_classifier, fscore_classifier, support_classifier =  precision_recall_fscore_support(yvalid, y_model)
        precisions.append(precision_classifier)
        recalls.append(recall_classifier)
        fscores.append(fscore_classifier)
        supports.append(support_classifier)
        estimators.append((model_strings[i], model))
        eclf = VotingClassifier(estimators = estimators, voting = 'hard')
    eclf.fit(Xtrain_scale, ytrain)
    y_model = eclf.predict(Xvalid_scale)
    percent_corr.append(accuracy_score(yvalid, y_model))
    confusion_matrices.append(confusion_matrix(yvalid, y_model))
    precision_classifier, recall_classifier, fscore_classifier, support_classifier =  precision_recall_fscore_support(yvalid, y_model)
    precisions.append(precision_classifier)
    recalls.append(recall_classifier)
    fscores.append(fscore_classifier)
    supports.append(support_classifier)
    return X, y, percent_corr, confusion_matrices, precisions, recalls

def run_feature_ranking(num_train):
    train, valid, test = io.read_MNIST()
    # Use train_test_split to pick a random sampling of the training and validation sets
    train1, train2, train1_labels, train2_labels = train_test_split(train[0], train[1], train_size = num_train)
    digit_imgs_train_np = [np.array(x).reshape(28,28) for x in train1]
    digit_imgs_train_np_w_rot = io.MNIST_add_rot(digit_imgs_train_np, 4, 15)
    ytrain = train1_labels
    digit_train_proc = sp.run_image_preprocess_MNIST(digit_imgs_train_np_w_rot)
    Xtrain = fe.features_MNIST(digit_train_proc)
    Xtrain_scale = scale(Xtrain, axis = 0)
    visualizer1D = Rank1D(algorithm = 'shapiro')
    visualizer1D.fit(Xtrain_scale, ytrain)
    visualizer1D.transform(Xtrain_scale)
    visualizer1D.poof()
    visualizer2D_cv = Rank2D(algorithm = 'covariance')
    visualizer2D_cv.fit(Xtrain_scale, ytrain)
    visualizer2D_cv.transform(Xtrain_scale)
    visualizer2D_cv.poof()
    visualizer2D_p = Rank2D(algorithm = 'pearson')
    visualizer2D_p.fit(Xtrain_scale, ytrain)
    visualizer2D_p.transform(Xtrain_scale)
    visualizer2D_p.poof()
    return digit_imgs_train_np, digit_imgs_train_np_w_rot

def run_fine_tune_classification(num_train, num_valid):
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
    clf = KNeighborsClassifier()

    param_grid_KNN = {"n_neighbors": [5,7,9,11,13,15,17,19],
                      "weights": ['uniform', 'distance'],
                      "algorithm": ['auto']}
                      #,'ball_tree','kd_tree', 'brute'
    grid_search = GridSearchCV(clf, param_grid = param_grid_KNN)
    grid_search.fit(Xtrain_scale, ytrain)
    report(grid_search.cv_results_)
    return

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def run_manifold_projection(n_comps, feature_mx, label_vec):
    iso = Isomap(2, n_components = n_comps)
    projection = iso.fit_transform(np.hstack((feature_mx, label_vec.reshape((label_vec.shape[0], 1)))))
    plt.close('all')

    if n_comps == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.scatter(projection[:,0], projection[:,1], c = label_vec, cmap = plt.cm.get_cmap('tab10', len(label_vec)), facecolor = 'none')
    elif n_comps ==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        p = ax.scatter(projection[:,0], projection[:,1], projection[:,2],  c = label_vec, cmap = plt.cm.get_cmap('tab10', len(label_vec)), facecolor = 'none')
    fig.colorbar(p, ticks = range(10), label = 'digit')
    plt.show()
