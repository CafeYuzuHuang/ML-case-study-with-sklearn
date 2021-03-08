"""
Apply linear and non-linear regression methods such as neural network and
support vector machine within the k-fold cross validation scheme to predict
the load curve of PS or thinfilm upon flat-end indentation.

2021.03.08: simplified version
"""
 
import datetime as dt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn.preprocessing as skpp
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import sklearn.metrics as skm
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV # exhaustive search (brute force)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

## Global variables
# (1) Constants
Tiny = 1e-4
OptIterMax = 500

# (2) Data source:
xlsxInFName = '20200610_PSML_DB_simple.xlsx' # Modify data info of group AHb
DBSheetName = 'DB_simple'

# (3) Settings:
Fea = ['Mask_open_type', 'PSH', 'Top95', 'Top90', 'Bot20', 'Bot10', 'Bot0', \
       'Cycles', 'Loadrate', 'Maxload', 'Minload', 'Holdtime', \
       'Formulation', 'B1', 'O1', 'O2', 'O3', 'M1', 'M2', 'M3', \
       'Others', 'Unknown']
Tar = ['Disp', 'Load', 'GKF'] # Tar[1] is not used during ML, Tar[2] as ylabel
Origin_col = ['RRID', 'Time_max']
Replaced_col = ['TestID', 'Cur_time']
Time_lag = 0.35 # lag time in load-unload test, should be less than T_step
T_step = 0.5 # time increment
T_max = 100 # max logged test time
Use_PS_Comp = True # if apply Nv-based PS composition
Fea_PS = ['B1', 'O1', 'O2', 'O3', 'M1', 'M2', 'M3', 'Others', 'Unknown']
cv_folds = 5
cv_chosen = 2
AssignedScore = {'nMAE': 'neg_mean_absolute_error',
                 'nMSE': 'neg_mean_squared_error',
                 'RSQ': 'r2'} # a dict (note: RMSE is not available)
SetGridRefit = True
# DefaultMetric = 'r2' # For fine tuning
DefaultMetric = 'neg_mean_squared_error' # Used if r^2 < 0 may occur
if len(AssignedScore) > 1:
    # Choose one of the metrics for refitting the best condition:
    # SetGridRefit = 'RSQ' # For fine tuning
    SetGridRefit = 'nMSE' # For rough tuning (since negative r^2 may occur)
    DefaultMetric = AssignedScore[SetGridRefit]
# To lookup valid scoring values, use the following statements:
# sorted(skm.SCORERS.keys())

# (4) Hyper-parameters for DNN machine learning:
"""
Default setting:
    Activation function: 'relu'
    Solver: 'adam'
    Mini batch size: < 250
    Hyperparameters for adam solver: use default
    Hyperparameters for sgd solver: use default
"""
# HLNeurons = (10, 10, 10) # default: (100, ) (One hidden layer w/ 100 neurons)
HLNeurons = (20, 10, 5, 5, 10, 20)
# HLNeurons = (5, 10, 20, 20, 10, 5)
AcFun = 'relu' # default: 'relu', activation function of hidden layers
Sol = 'adam' # default: 'adam', solver
Alpha = Tiny # default: 1e-4, L2-norm parameter
MiniBS = 200 # default: 'auto' = min(200, n_samples), mini-batch size
MaxIt = OptIterMax # default: 200, max # of iterations
RandS = 87 # default = None, random state
TolErr = Tiny # default: 1e-4 tolerance
EStop = False # default: False, early stopping
# If True, default validation fraction = 0.1 (non-group-type train-valid split)
ESVF = 0.1 # default = 0.1, bounds (0, 1), validation fraction for early stopping
Shuf = True # default = True, whether to shuffle samples in each iteration
## If solver = 'sgd', the following hyperparameters should be set appropriately
## Use default values may be sufficient
LearnRate = 'constant' # default = 'constant', learning rate
LRI = 0.001 # default = 0.001, initial learning rate
MTM = 0.9 # default = 0.9, bound = (0, 1), used in sgd solver only
NestM = True # default = True, Nesterov¡¦s momentum, used in sgd solver
## If solver = 'adam', some hyperparameters available for fine tuning
## However, use default values are sufficient

# Rough tuning via random search (Case 3):
N0 = 20 # recommended: # of feature input
N1 = int(N0/2)
N2 = int(N1/2)
N3 = int(N2/2)
H1 = (N0, N1, N2, N3)
H2 = (N3, N2, N1, N0)
H3 = (N0, N0, N0, N0)
H4 = (N0, N0, N0, N0, N0, N0, N0, N0)
H5 = (N0, N1, N0, N1, N0, N1, N0, N1)
H6 = (N0, N1, N2, N3, N3, N2, N1, N0)
H7 = (N3, N2, N1, N0, N0, N1, N2, N3)
H8 = (N0, N1, N0, N1, N0, N1, N2, N3)
param_rs = {"mlp__alpha": [1e-4, 1e-2, 1e-0], \
            "mlp__hidden_layer_sizes": [H1, H2, H3, H4, H5, H6, H7, H8], \
            "mlp__early_stopping": [True, False]}
nsamples = 12 # default = 10, here set # samples equals to that in Case 4
# Fine tuning via grid search (Case 4): 12 tests in total
MaxIt = 15000 # If sgd solver is used, larger max_iter may be required!
param_gs = {"mlp__alpha": [1e-4, 1e-2, 1e-0], \
            "mlp__solver": ['adam', 'sgd'], \
            "mlp__early_stopping": [True, False]}

def VisualizeDNN(dnn_coef, in_name, out_name):
    """
    Deep neural network visualization
    Info displayed:
        (1) Name of input(s) and output(s) (text)
        (2) # of neurons in each layer (text)
        (3) Coefficients (weights) (colored lines)
        (4) Biases of layers are not shown in the current version
    """
    # Initialization
    neurons_each_layer = []
    neurons_coor_x = []
    neurons_coor_y = []
    in_txt_coor_x = []
    in_txt_coor_y = []
    out_txt_coor_x = []
    out_txt_coor_y = []
    hid_txt_coor_x = []
    hid_txt_coor_y = []
    
    # Get info
    n_connects_layer = len(dnn_coef)
    n_h_layers = n_connects_layer - 1
    for ii in range(0, n_connects_layer):
        # Get # of inputs and neurons of each hidden layer
        neurons_each_layer.append(len(dnn_coef[ii]))
    neurons_each_layer.append(len(dnn_coef[n_h_layers][0])) # # of outputs
    max_neurons = max(neurons_each_layer)
    # min_neurons = min(neurons_each_layer)
    
    # Assign draw parameters
    dx_layers = 100.0 # distance between layers
    dy_neurons = 0.6*dx_layers # distance between neurons
    dx_space = 2.0*dx_layers # spacing
    dy_space = 2.0*dx_layers # spacing
    txt_space = 0.8*dx_layers # spacing
    r_neuron = 0.25*dx_layers # radius of each neuron
    x_max = dx_layers*n_connects_layer + dx_space*2.0
    y_max = dy_neurons*(max_neurons - 1.0) + dy_space*2.0
    Lwidth = 1 # connection line width
    
    # Assign coordinates of each neuron
    x_layer_cur = dx_space # current x position
    y_mid = 0.5*y_max # midpoint of y position
    for ii in range(0, len(neurons_each_layer)):
        n = neurons_each_layer[ii]
        x_coor = []
        y_coor = []
        tmp_y = []
        if n % 2 == 1: # odd value
            # e.g. if 7 neurons -> 3, 2, 1, 0, -1, -2, -3
            y_start = n // 2
            y_end = -1*(n // 2) - 1
            tmp_y = np.arange(y_start, y_end, -1)
        else: # even value
            # e.g. if 8 neurons -> 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5
            y_start = (n // 2) - 0.5
            y_end = -1*(n // 2) - 0.5
            tmp_y = np.arange(y_start, y_end, -1)   
        for jj in range(0, len(tmp_y)):
            x_coor.append(x_layer_cur)
            y_cur = y_mid + float(tmp_y[jj])*dy_neurons
            y_coor.append(y_cur)
        neurons_coor_x.append(x_coor) # collect x coordinates of neurons
        neurons_coor_y.append(y_coor) # collect y coordinates of neurons
        
        if ii == 0: # input layer
            in_txt_coor_x = [x_layer_cur - txt_space]*neurons_each_layer[ii]
            in_txt_coor_y = y_coor
        elif ii == n_connects_layer: # output layer
            out_txt_coor_x = [x_layer_cur + txt_space]*neurons_each_layer[ii]
            out_txt_coor_y = y_coor
        else: # hidden layers
            hid_txt_coor_x.append(x_layer_cur)
            hid_txt_coor_y.append(max(y_coor) + txt_space)
        x_layer_cur += dx_layers # go to next layer
    
    # Start plotting:
    plt.figure(figsize = (8, 6))
    ax = plt.gca() # gca means "get current axes"
    # Deal with neurons
    for ii in range(0, len(neurons_each_layer)):
        for jj in range(0, neurons_each_layer[ii]):
            x = neurons_coor_x[ii][jj]
            y = neurons_coor_y[ii][jj]
            DNN_Nodes = plt.Circle((x, y), r_neuron, ec = 'k', fc = 'w', \
                                   zorder = 4) # higher zorder draws later
            ax.add_artist(DNN_Nodes)
    # Deal with links
    # Get the RGBA values from a float, the shape of list is i x j x k x 4
    colors = [cm.coolwarm(color) for color in dnn_coef]
    dnn_flatten = [] # Flatten dnn_coef: used later
    for ii in range(0, n_connects_layer):
        # Connection network between layer ii and ii+1:
        for jj in range(0, neurons_each_layer[ii]):
            for kk in range(0, neurons_each_layer[ii+1]):
                xj = neurons_coor_x[ii][jj]
                yj = neurons_coor_y[ii][jj]
                xk = neurons_coor_x[ii+1][kk]
                yk = neurons_coor_y[ii+1][kk]
                DNN_Edges = plt.Line2D([xj, xk], [yj, yk], linewidth = Lwidth, \
                                       c = colors[ii][jj][kk], zorder = 1)
                ax.add_artist(DNN_Edges)
                dnn_flatten.append(dnn_coef[ii][jj][kk])
    
    # Deal with labels
    for ii in range(0, len(neurons_each_layer)):
        if ii == 0: # input layer
            for jj in range(0, neurons_each_layer[ii]):
                plt.text(in_txt_coor_x[jj], in_txt_coor_y[jj], in_name[jj], \
                         color = 'g', ha = 'right', va = 'center')           
        elif ii == n_connects_layer: # output layer
            for jj in range(0, neurons_each_layer[ii]):
                plt.text(out_txt_coor_x[jj], out_txt_coor_y[jj], out_name[jj], \
                         color = 'g', ha = 'left', va = 'center')           
        else: # hidden layers
            plt.text(hid_txt_coor_x[ii-1], max(hid_txt_coor_y), \
                     str(neurons_each_layer[ii]), color = 'g', \
                     ha = 'center', va = 'center')
    
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])
    ax.set_aspect(1.0)
    plt.scatter(dnn_flatten, dnn_flatten, s = 0, c = dnn_flatten, \
                cmap = 'coolwarm')
    plt.colorbar() # Colorbar is the thing what we need here
    plt.axis('off')
    plt.show() # Done!
    # No return


def FeatureOHE(df0):
    """
    Preprocessing feature matrix by:
        (1) standardized scaling if numeric columns (after train-test splitting)
        (2) one-hot encoding if categorical columns (before train-test splitting)
        (3) for PS formulation, two modes applicable:
            (i) one-hot encoding for unknown PS only (Use_PS_Comp = True)
            (ii) one-hot encoding for all PS (Use_PS_Comp = False)
    """
    # Convert all categorical columns into one-hot encoding (OHE)
    # After conversion, categorical columns are dropped automatically
    OHE = pd.get_dummies(df0, drop_first = False, \
                         columns = ['Mask_open_type', 'Formulation'], \
                         dummy_na = False, prefix_sep = '_')
    if Use_PS_Comp == False:
        # Drop columns listed in Fea_PS
        OHE.drop(columns = Fea_PS, inplace = True)
        print("\n\nmyPSML_202005.FeatureOHE() checks: ")
        print("Whether Nv-based PS comp. is applied: ", Use_PS_Comp)
        print("Size of feature matrix after OHE: ", OHE.shape)
    else:
        # Drop columns of formulation names with known composition (Unknown = 0)
        ohe1 = pd.get_dummies(df0[df0['Unknown'] > 0], drop_first = False, \
                              columns = ['Mask_open_type', 'Formulation'], \
                              dummy_na = False)
        PSList_All = OHE.columns[OHE.columns.str.contains(pat = 'Formulation_')]
        PSList_Unknown = ohe1.columns[ohe1.columns.str.contains(pat = 'Formulation_')]
        PSList_Known = [item for item in PSList_All if item not in PSList_Unknown]
        OHE.drop(columns = PSList_Known, inplace = True)
        OHE.drop('Unknown', axis = 1, inplace = True)
    OHE.drop('TestID', axis = 1, inplace = True)
    # Export to .xlsx file
    OHE.to_excel('myPSML_202005_FeatureOHE.xlsx', \
                 sheet_name = 'Encoded features', index = True)
    print("myPSML_202005.FeatureOHE() finished!")
    return OHE

def TableTransform(df0):
    """
    Features:
        Entry
        TestID: df0['Group'] + '-' + df0['GID'].astype(str) (or apply RRID)
        PSH & CDs
        RR test conditions
        PS composition, NV-based
        current time
    The feature table will look like:
        Entry TestID PSH ... B1 ... time
        1    A-1-1    a1    b1    0.5
        2    A-1-1    a1    b1    1.0
        3    A-1-1    a1    b1    1.5
        ...
        n-1  A-1-1    a1    b1    t1-max
        n    A-2-1    a2    b2    0.5
        ...
        m    Z-k-l    ax    bx    tx
    The target table will look like:
        Entry TestID    time    disp.
        1    A-1-1    0.5    z1-1
        2    A-1-1    1.0    z1-2
        3    A-1-1    1.5    z1-3
        ...
        n-1  A-1-1    t1-max    z1-max
        n    A-2-1    0.5    z2-1
        ...
        m    Z-k-l    tx    zx-n
    """
    dfx = pd.DataFrame(columns = Replaced_col + Fea)
    dfy = pd.DataFrame(columns = Replaced_col + Tar)
    ts = pd.Series(data = np.arange(0, T_max, T_step))
    ind = 1 # index = counter
    for ii in range (0, df0.shape[0]):
        # RRID is replaced by TestID; then, Time_max is replaced by Cur_time
        # Load is calculated from RR conditions and current time
        tmax = df0.loc[df0.index[ii], Origin_col[1]] # should be less than T_max
        max_jj = int(tmax/T_step) # should be less than (T_max/T_step + 1)
        TL = Time_lag
        LR = df0.loc[df0.index[ii], 'Loadrate']
        FM = df0.loc[df0.index[ii], 'Maxload']
        Fm = df0.loc[df0.index[ii], 'Minload']
        HT = df0.loc[df0.index[ii], 'Holdtime']
        C = df0.loc[df0.index[ii], 'Cycles']
        for jj in range(1, max_jj):
            # Construct feature table
            dfx.loc[ind] = df0.loc[df0.index[ii], Fea]
            dfx.loc[ind, Replaced_col[0]] = df0.loc[df0.index[ii], Origin_col[0]]
            dfx.loc[ind, Replaced_col[1]] = ts.loc[jj]
            # Construct target table
            dfy.loc[ind, Tar[0]] = df0.loc[df0.index[ii], ts.loc[jj]]
            dfy.loc[ind, Tar[1]] = GetLoad(ts.loc[jj], TL, LR, FM, Fm, HT, C)
            # dfy.loc[ind, Tar[2]] = Labels[ii] # Group ID as group cv label
            dfy.loc[ind, Tar[2]] = df0.index[ii] # RR test ID as group cv label
            dfy.loc[ind, Replaced_col[0]] = df0.loc[df0.index[ii], Origin_col[0]]
            dfy.loc[ind, Replaced_col[1]] = ts.loc[jj]
            ind += 1
    print("myPSML_202005.TableTransform() checks # of rows: ", ind)
    print("Shape of feature table: ", dfx.shape)
    print("Shape of target table: ", dfy.shape)
    # Export to .xlsx file
    with pd.ExcelWriter('myPSML_202005_TableTransform.xlsx') as writer: 
        dfx.to_excel(writer, sheet_name = 'unscaled features', index = True)
        dfy.to_excel(writer, sheet_name = 'targets and labels', index = True)
    print("myPSML_202005.TableTransform() finished!")
    return dfx, dfy

# Called by TableTransform()
def GetLoad(t, TL, LR, FM, Fm, HT, C):
    """
    Calculate current load
    Input arguments are: current time, time lag, load rate (= negative unload rate),
    max load, min load, hold time (at max and min load), cycles of test
    """
    # Validation:
    if TL < 0.0 or LR < 0.0 or FM < 0.0 or Fm < 0.0 or HT < 0.0 or C < 1.0:
        print("Warning: myPSML_202005.GetLoad() found invalid arguments!")
        print("Examine input arguments (t, TL, LR, FM, Fm, HT, C): ")
        print(t, TL, LR, FM, Fm, HT, C)
        print("Return negative load (-111) and exit function ...")
        return -111.0
    if t < 0.0 or t < TL or FM < Fm:
        print("Warning: myPSML_202005.GetLoad() found invalid arguments!")
        print("Examine input arguments (t, TL, LR, FM, Fm, HT, C): ")
        print(t, TL, LR, FM, Fm, HT, C)
        print("Return negative load (-222) and exit function ...")
        return -222.0
    # Determine load:
    tpre = Fm/LR + TL # preload time
    tcy = ((FM - Fm)/LR + HT)*2.0 # period per cycle
    tend = tpre + tcy*C # max time
    if C >= 1.0:
        if t < tend:
            if t < tpre:
                return (t - TL)*LR
            else:
                cur_phase = ((t - tpre)%tcy)/tcy
                if cur_phase < (0.5 - HT/tcy): # load stage
                    return Fm + cur_phase/(0.5 - HT/tcy)*(FM - Fm)
                elif cur_phase <= 0.5: # hold at max load
                    return FM
                elif cur_phase < (1.0 - HT/tcy): # unload
                    return FM - (cur_phase - 0.5)/(0.5 - HT/tcy)*(FM - Fm)
                else: # hold at min load
                    return Fm
        else: # t > tend, output trivial value = min load
            return Fm
    else:
        # unexpected to enter here
        print("Warning: myPSML_202005.GetLoad() found unexpected problem!")
        print("Examine input arguments (t, TL, LR, FM, Fm, HT, C): ")
        print(t, TL, LR, FM, Fm, HT, C)
        print("Return negative load (-999) and exit function ...")
        return -999.0

## main()
if __name__ == '__main__': # script run of this file (for function testing)
    print("=================================================")
    print("In myPSML_202005: check input data filename: ", xlsxInFName)
    
    # Timer setting
    t_start = dt.datetime.now()
    print("\n\nmyPSML_202005.main() shows start time: ", t_start)
    
    # Load dataset
    df = pd.read_excel(xlsxInFName, sheet_name = DBSheetName, index_col = 0)
    print("Load sheetname: ", DBSheetName)
    print("Matrix size: ", df.shape)
    
    ## Preprocessing: shared by all cases
    # Define features and targets
    test_df0 = df.loc[df.index <= 85] 
    print("Check matrix size of test dataframe: ", test_df0.shape)
    X_0, Y_0 = TableTransform(test_df0) # For test
    
    # Encoding for categorical features; scaling for numerics
    # encoding -> data splitting -> training set and testing set scaling
    Xohe_0 = FeatureOHE(X_0) # One-hot encoding for categorical features
    
    # Drop columns with single level or zero stdev
    ooo = [item for item in Xohe_0.columns if len(Xohe_0[item].unique()) > 1]
    ppp = [item for item in Xohe_0.columns if item not in ooo]
    Xohe = Xohe_0.loc[:, ooo]
    print("Check column list after one-hot encoding: ", Xohe_0.columns)
    print("Column list after removing single-level columns: ", Xohe.columns)
    print("Columns removed: ", ppp)
    print("# of columns: ", len(Xohe_0.columns), len(Xohe.columns), len(ppp))
    # Export to .xlsx file
    with pd.ExcelWriter('myPSML_202005_FeatureOHE_updated.xlsx') as writer: 
        Xohe_0.to_excel(writer, sheet_name = 'Encoded features', index = True)
        Xohe.to_excel(writer, sheet_name = 'No single level columns', \
                      index = True)
    
    # Train-test splitting: use group K-fold; test set for prediction
    gkf_outer = GroupKFold(n_splits = cv_folds) # default n_splits = 5
    gkf = GroupKFold(n_splits = cv_folds) # default n_splits = 5
    ylabel = Y_0[Tar[2]] # by test #
    yexp = Y_0[Tar[0]] # displacement at each t
    
    cv_cur = 1 # current cv fold
    for train_ind, test_ind in gkf_outer.split(X = Xohe, y = yexp, \
                                               groups =  ylabel):
        if cv_cur == cv_chosen: # replace indices
            train_index, test_index = train_ind, test_ind
        cv_cur += 1
    
    X_train = Xohe.loc[Xohe.index[train_index]]
    X_test = Xohe.loc[Xohe.index[test_index]]
    y_train = yexp.loc[yexp.index[train_index]]
    y_test = yexp.loc[yexp.index[test_index]]
    g_train = ylabel.loc[ylabel.index[train_index]] # for train-validation split
    g_test = ylabel.loc[ylabel.index[test_index]]
    
    # Skip one-hot encoded columns during standard scaling
    PSList_all = Xohe.columns[Xohe.columns.str.contains(pat = 'Formulation_')]
    MOT_all = Xohe.columns[Xohe.columns.str.contains(pat = 'Mask_open_type_')]
    OHE_all = np.concatenate([PSList_all, MOT_all], axis = 0)
    Num_col = [item for item in Xohe.columns if item not in OHE_all]
    prep = make_column_transformer((skpp.StandardScaler(), Num_col), \
                                   remainder='passthrough')
    
    # Hyperparameter search
    # Use randomized search with wider ranges of parameter set
    mlp = MLPRegressor(hidden_layer_sizes = HLNeurons, activation = AcFun, \
                       solver = Sol, alpha = Alpha, batch_size = MiniBS, \
                       max_iter = MaxIt, random_state = RandS, tol = TolErr, \
                       early_stopping = EStop, validation_fraction = ESVF, \
                       shuffle = Shuf, learning_rate = LearnRate, \
                       learning_rate_init = LRI, momentum = MTM, \
                       nesterovs_momentum = NestM)
    pipe_dnn = Pipeline([('scaler', prep), ('mlp', mlp)])
    
    dnn = RandomizedSearchCV(estimator = pipe_dnn, cv = gkf, \
                             scoring = AssignedScore, refit = SetGridRefit, \
                             return_train_score = True, \
                             param_distributions = param_rs,\
                             n_iter = nsamples, random_state = None)
    dnn.fit(X = X_train.astype(np.float64), y = y_train, groups = g_train)
    
    print("Show results: \n", dnn.cv_results_)
    print("\n=================================================\n")
    print("Best parameters: ", dnn.best_params_)
    print("# of CV folds: ", dnn.n_splits_)
    print("Best mean CV score: ", dnn.best_score_) # validation set
    
    # Visualize DNN architecture:
    print("\nNow visualize DNN weights of the searched best one: ")
    VisualizeDNN(dnn.best_estimator_.named_steps.mlp.coefs_, \
                 X_train.columns, [y_train.name])
    
    # Show loss vs. epoch
    print("Loss function type: ", dnn.best_estimator_.named_steps.mlp.loss)
    print("Loss of DNN: ", dnn.best_estimator_.named_steps.mlp.loss_)
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.plot(dnn.best_estimator_.named_steps.mlp.loss_curve_)
    plt.title("Grid search with cross validation and best parameter set")
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    XlsxOutFName = 'myPSML_202005_4th_pred_case3.xlsx'
    
    ## Postprocessing (model diagnosis)
    # Note that the best_estimator may be refit by each CV-related functions
    # Check CV (validation set) score:
    CVS = cross_val_score(dnn.best_estimator_, X = X_train.astype(np.float64), \
                          y = y_train, groups = g_train, \
                          scoring = DefaultMetric, cv = gkf)
    print("\n=================================================\n")
    print("Cross-validation analysis (training+validation):")
    print("CV scoring metric: ", SetGridRefit)
    print("CV scores (validation set): ", CVS)
    print("Mean CV score: ", CVS.mean())
    print("Stdev of CV score: ", CVS.std())
    print("Recall best mean CV score in grid search: ", dnn.best_score_)
    
    # The following two function calls give same results:
    CVS1 = cross_val_score(dnn.best_estimator_, X = Xohe.astype(np.float64), \
                           y = yexp, groups = ylabel, \
                           scoring = DefaultMetric, cv = gkf_outer)
    print("\nCross-validation analysis (all data):")
    print("CV scoring metric: ", SetGridRefit)
    print("CV scores (test set): ", CVS1)
    print("Mean CV score: ", CVS1.mean())
    print("Stdev of CV score: ", CVS1.std())
    print("Recall best mean CV score in grid search: ", dnn.best_score_)
    
    # Check and plotting learning curve: (use all data set and test again)
    # Applied to find converged training data size and thus avoid underfitting
    # We found sklearn.learning_curve will reserve testing set, then dividing
    # the training and validation sets
    ts_frac_ticks = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = \
    learning_curve(dnn.best_estimator_, X = Xohe.astype(np.float64), \
                   y = yexp, groups = ylabel, scoring = DefaultMetric, \
                   train_sizes = ts_frac_ticks, cv = gkf_outer)
    print("\n\nmyPSML_202005.main() checks learning curve ...")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    Ystr = "Metric score: " + DefaultMetric
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", \
             label="Training set")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", \
             label="Test set")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, \
                     train_scores_mean + train_scores_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, \
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.xlabel("Training size")
    plt.ylabel(Ystr)
    plt.title("Learning curve")
    plt.grid(which = 'both', axis = 'both')
    plt.legend(loc = "best")
    plt.show()
    print("Total data set size: ", Xohe.shape[0])
    print("Training + validation size (fraction): \n", ts_frac_ticks)
    print("Training + validation size (roughly estimated): \n", \
          ts_frac_ticks*Xohe.shape[0]*(1-1/cv_folds))
    print("Training + validation size (real values): \n", train_sizes)
    print("Training score mean: \n", train_scores_mean)
    print("Training score stdev: \n", train_scores_std)
    print("Test score mean: \n", test_scores_mean)
    print("Test score stdev: \n", test_scores_std)
    
    ## Postprocessing (result analysis) 
    print("\n\nMultiple scoring: check train+valid scores and test scores:")
    disp_fit = dnn.best_estimator_.predict(X = X_train.astype(np.float64))
    disp_pred = dnn.best_estimator_.predict(X = X_test.astype(np.float64))
    print("Training + validation set: CV fold = ", cv_chosen)
    print("Mean absolute error: ", skm.mean_absolute_error(y_train, disp_fit))
    print("RMSE: ", math.sqrt(skm.mean_squared_error(y_train, disp_fit)))
    print("r2: ", skm.r2_score(y_train, disp_fit))
    print("\nTesting set: CV fold = ", cv_chosen)
    print("Mean absolute error: ", skm.mean_absolute_error(y_test, disp_pred))
    print("RMSE: ", math.sqrt(skm.mean_squared_error(y_test, disp_pred)))
    print("r2: ", skm.r2_score(y_test, disp_pred))
    
    print("\n\nEstimate with CV: check test scores of all CV folds:")
    y_cv_pred = cross_val_predict(dnn.best_estimator_, \
                                  X = Xohe.astype(np.float64), \
                                  y = yexp, groups = ylabel, cv = gkf_outer)
    print("Testing sets: # of CVs = ", cv_folds)
    print("Mean absolute error: ", skm.mean_absolute_error(yexp, y_cv_pred))
    print("RMSE: ", math.sqrt(skm.mean_squared_error(yexp, y_cv_pred)))
    print("r2: ", skm.r2_score(yexp, y_cv_pred))
    
    # Plotting:
    y_test_X = np.reshape(y_test.values, newshape = [y_test.shape[0], 1])
    reg = LinearRegression(fit_intercept = False)
    slope = reg.fit(X = y_test_X, y = disp_pred).coef_
    rsq = reg.score(X = y_test_X, y = disp_pred)
    XX = np.array([y_test.min(), y_test.max()])
    YY = slope*XX
    X0 = np.array([0.0, y_test.max()*1.25])
    STR = "Test set prediction of chosen fold, slope = " \
    + str(round(slope[0], 4)) + ", R^2 = " + str(round(rsq, 4))
    
    print("\n\nmyPSML_202005.main() now plots predicted results ...")
    # Plot test set only to reduce data points in the plot
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.scatter(y_test, disp_pred, c = 'b', marker = 'o', alpha = 0.5)
    plt.plot(XX, YY, color = 'r', linewidth = 2) # y_pred = slope*y_exp
    plt.plot(X0, X0, color = 'k', linewidth = 4) # X = Y line
    plt.xlabel("Expected disp. (um)")
    plt.ylabel("Predicted disp. (um)")
    plt.title(STR)
    plt.xlim(X0[0], X0[1])
    plt.ylim(X0[0], X0[1])
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    # Plotting:
    y_all_X = np.reshape(yexp.values, newshape = [yexp.shape[0], 1])
    reg = LinearRegression(fit_intercept = False)
    slope = reg.fit(X = y_all_X, y = y_cv_pred).coef_
    rsq = reg.score(X = y_all_X, y = y_cv_pred)
    XX = np.array([yexp.min(), yexp.max()])
    YY = slope*XX
    X0 = np.array([0.0, yexp.max()*1.25])
    STR = "Test set prediction with all CV folds, slope = " \
    + str(round(slope[0], 4)) + ", R^2 = " + str(round(rsq, 4))
    
    print("\n\nmyPSML_202005.main() now plots predicted results ...")
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.scatter(yexp, y_cv_pred, c = 'b', marker = 'o', alpha = 0.5)
    plt.plot(XX, YY, color = 'r', linewidth = 2) # y_pred = slope*y_exp
    plt.plot(X0, X0, color = 'k', linewidth = 4) # X = Y line
    plt.xlabel("Expected disp. (um)")
    plt.ylabel("Predicted disp. (um)")
    plt.title(STR)
    plt.xlim(X0[0], X0[1])
    plt.ylim(X0[0], X0[1])
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    # Export results to excel:
    print("\n\nmyPSML_202005.main() now exports predicted results ...")
    disp_f_s = pd.Series(disp_fit, index = y_train.index, name = 'DNN')
    disp_p_s = pd.Series(disp_pred, index = y_test.index, name = 'DNN')
    y_cv_p_s = pd.Series(y_cv_pred, index = yexp.index, name = 'DNN')
    fit_df = pd.concat([g_train, X_train, y_train, disp_f_s], axis = 1)
    pred_df = pd.concat([g_test, X_test, y_test, disp_p_s], axis = 1)
    cv_p_df = pd.concat([ylabel, Xohe, yexp, y_cv_p_s], axis = 1)
    with pd.ExcelWriter(XlsxOutFName) as writer: 
        fit_df.to_excel(writer, sheet_name = 'train and valid', index = True)
        pred_df.to_excel(writer, sheet_name = 'test set pred', index = True)
        cv_p_df.to_excel(writer, sheet_name = 'CV pred all', index = True)
    
    ## Ending
    # Timer setting
    t_end = dt.datetime.now()
    print("\n\nmyPSML_202005.main() shows finish time: ", t_end)
    print("Total ellapsed time is: ", t_end - t_start)
    print("Finish myPSML_202005! Now exiting ...")
    print("=================================================")
# Done!!

