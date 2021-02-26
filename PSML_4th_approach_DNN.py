"""
Apply linear and non-linear regression methods such as neural network and
support vector machine within the k-fold cross validation scheme to predict
the load curve of PS or thinfilm upon flat-end indentation.

Version log: Updated on Jun. 19, 2020
=========================================================================
ver.        date         features
=========================================================================
0.0        2020.06.19    New release
                         Case 3 & 4 validated (not optimized)
                         VisualizeDNN() implemented
                         Case 1 & 2 validated (not optimized)
"""
 
import datetime as dt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.optimize as opt
import matplotlib.cm as cm

## sci-kit learn modules and functions applied:
import sklearn.preprocessing as skpp
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import sklearn.metrics as skm
from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV # exhaustive search (brute force)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
# from sklearn.svm import SVR # Epislon-supporting vector machines
# from sklearn.svm import NuSVR # Nu-supporting vector machines
from sklearn.linear_model import LinearRegression

## sci-kit learn modules and functions not applied here:
# from sklearn.model_selection import SelectKBest
# from sklearn.model_selection import train_test_split # Applied before CV
# from sklearn.model_selection import RepeatedKFold
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression # for CV-PSLR

# %matplotlib inline

## Global variables
# (1) Constants
Tiny = 1e-4
# Tiny2 = 1e-8
# Tiny3 = 1e-12
OptIterMax = 500

# (2) Data source:
xlsxInFName = '20200610_PSML_DB_simple.xlsx' # Modify data info of group AHb
# xlsxInFName = '20200521_PSML_DB.xlsx'
DBSheetName = 'DB_simple'

# (3) Settings:
# Group_pred = ['AHa', 'AHb', 'AI'] # excluded in model training & validation
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
# cv_folds = 3
cv_chosen = 2
AssignedScore = {'nMAE': 'neg_mean_absolute_error',
                 'nMSE': 'neg_mean_squared_error',
                 'RSQ': 'r2'} # a dict (note: RMSE is not available)
SetGridRefit = True
# DefaultMetric = 'r2' # For fine tuning
DefaultMetric = 'neg_mean_squared_error' # Used if r^2 < 0 may occur
if len(AssignedScore) > 1:
    # SetGridRefit = False
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
    # print("x_max and y_max: ", x_max, y_max)
    
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
        # print("myPSML_202005.VisualizeDNN() checks current layer: ", ii)
        # print("x coordinates: ", x_coor)
        # print("y coordinates: ", y_coor, "\nMax value = ", max(y_coor))
        
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
    # plt.figure() # default figsize = 6.4 x 4.8 (inches)
    plt.figure(figsize = (8, 6))
    ax = plt.gca() # gca means "get current axes"
    # Deal with neurons
    # print("myPSML_202005.VisualizeDNN() starts plotting neurons:")
    for ii in range(0, len(neurons_each_layer)):
        for jj in range(0, neurons_each_layer[ii]):
            x = neurons_coor_x[ii][jj]
            y = neurons_coor_y[ii][jj]
            DNN_Nodes = plt.Circle((x, y), r_neuron, ec = 'k', fc = 'w', \
                                   zorder = 4) # higher zorder draws later
            ax.add_artist(DNN_Nodes)
    # Deal with links
    # Get the RGBA values from a float, the shape of list is i x j x k x 4
    # print("myPSML_202005.VisualizeDNN() starts plotting links:")
    colors = [cm.coolwarm(color) for color in dnn_coef]
    # print("myPSML_202005.VisualizeDNN() finishes color setting!")
    dnn_flatten = [] # Flatten dnn_coef: used later
    for ii in range(0, n_connects_layer):
        # Connection network between layer ii and ii+1:
        for jj in range(0, neurons_each_layer[ii]):
            for kk in range(0, neurons_each_layer[ii+1]):
                xj = neurons_coor_x[ii][jj]
                yj = neurons_coor_y[ii][jj]
                xk = neurons_coor_x[ii+1][kk]
                yk = neurons_coor_y[ii+1][kk]
                # wijk = dnn_coef[ii][jj][kk] # weight of the connection
                DNN_Edges = plt.Line2D([xj, xk], [yj, yk], linewidth = Lwidth, \
                                       c = colors[ii][jj][kk], zorder = 1)
                ax.add_artist(DNN_Edges)
                dnn_flatten.append(dnn_coef[ii][jj][kk])
    
    # Deal with labels
    # print("myPSML_202005.VisualizeDNN() starts plotting texts:")
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
            # tmp_str = str(neurons_each_layer[ii]) + ' neurons'
            # plt.text(hid_txt_coor_x[ii-1], hid_txt_coor_y[ii-1], tmp_str, \
            #          color = 'g', ha = 'center', va = 'center')
            plt.text(hid_txt_coor_x[ii-1], max(hid_txt_coor_y), \
                     str(neurons_each_layer[ii]), color = 'g', \
                     ha = 'center', va = 'center')
    
    # Use scatter plot:
    # The datapoints are hidden by set marker size = 0, thus (x, y) can be
    # arbitrarily assigned (shape of x and y should be i x j x k)
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])
    ax.set_aspect(1.0)
    plt.scatter(dnn_flatten, dnn_flatten, s = 0, c = dnn_flatten, \
                cmap = 'coolwarm')
    plt.colorbar() # Colorbar is the thing what we need here
    plt.axis('off')
    plt.show() # Done!
    # No return

"""
# 2020.06.19: Debug demo: 3 x 4 x 2
INNAME = ['In1', 'In2', 'In3']
OUTNAME = ['Out1', 'Out2']
DNNCOEF = [[[0.1, 0.2, -0.1, -0.5], [1.5, -1.2, 0.3, 2.0], [-2.0, -0.2, 0.1, 0.2]],
           [[0.2, -0.1], [1.2, -0.3], [2.2, -1.9], [-0.7, 0.5]]]
VisualizeDNN(DNNCOEF, INNAME, OUTNAME)
"""

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
    # le = skpp.LabelEncoder()
    # Labels = le.fit_transform(df0['Group']) # convert string to numeric labels
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
    test_df0 = df.loc[df.index <= 59] # Group AH~AI are excluded (default)
    # test_df0 = df.loc[df.index <= 85] # Group AHa, AHb, and AI are excluded
    # test_df0 = df.loc[(df['Group'] == 'A') | (df['Group'] == 'AH') | \
    #                   (df['Group'] == 'AHb')] # For debugging purpose
    # test_df0 = df.loc[df['Group'] == 'A'] # For debugging purpose
    # test_df0 = df.loc[df['Group'] == 'AI'] # For debugging purpose
    print("Check matrix size of test dataframe: ", test_df0.shape)
    X_0, Y_0 = TableTransform(test_df0) # For test
    
    # Encoding for categorical features; scaling for numerics
    # encoding -> data splitting -> training set and testing set scaling
    Xohe_0 = FeatureOHE(X_0) # One-hot encoding for categorical features
    
    # 2020.06.18 added: drop columns with single level or zero stdev
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
        # print("\n\nCurrent fold #: ", cv_cur)
        # print("TRAIN:", train_ind, "\nTEST:", test_ind)
        if cv_cur == cv_chosen: # replace indices
            train_index, test_index = train_ind, test_ind
        cv_cur += 1
        # Outputs of gkf_outer.split are:
        # A l x n array records indices of training sets of n-fold CV
        # A m x n array records indices of testing sets of n-fold CV
        # l ~ (1 - 1/n)*N, m ~ 1/n*N, where N is # of rows of data
        """
        X_train, X_test = \
        Xohe.loc[Xohe.index[train_ind], :], Xohe.loc[Xohe.index[test_ind], :]
        y_train, y_test = \
        Y_0.loc[Y_0.index[train_ind], :], Y_0.loc[Y_0.index[test_ind], :]
        """
    # print("\n\nSelected fold #: ", cv_chosen)
    # print("TRAIN:", train_index, "\nTEST:", test_index)
    
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
    
    """
    ## Machine learning of Case 1 and Case 2: (optimization and prediction)
    # Case 1: Uncertainty qualification (1)
    # Stdev of predicted results from DNNs by different train-valid splitting
    mlp = MLPRegressor(hidden_layer_sizes = HLNeurons, activation = AcFun, \
                       solver = Sol, alpha = Alpha, batch_size = MiniBS, \
                       max_iter = MaxIt, random_state = RandS, tol = TolErr, \
                       early_stopping = EStop, validation_fraction = ESVF, \
                       shuffle = Shuf, learning_rate = LearnRate, \
                       learning_rate_init = LRI, momentum = MTM, \
                       nesterovs_momentum = NestM)
    
    dnn = Pipeline([('scaler', prep), ('mlp', mlp)])
    
    # Obtain fit model by each CV fold
    print("Check training+validation set size: ", X_train.shape[0])
    print("Check testing set size: ", X_test.shape[0])
    cv_cur = 1
    y_test_pred = np.zeros([X_test.shape[0], cv_folds])
    y_test_score = np.zeros([3, cv_folds])
    dnn_lcs = []
    for train_ind, test_ind in gkf.split(X = X_train, y = y_train, \
                                               groups =  g_train):
        # test_ind (validation set) is not used
        X_t = X_train.loc[X_train.index[train_ind]]
        y_t = y_train.loc[y_train.index[train_ind]]
        print("CV fold # = ", cv_cur)
        print("Check training set size: ", len(train_ind))
        print("Check validation set size: ", len(test_ind))
        # g_t = g_train.loc[g_train.index[train_ind]]
        dnn.fit(X = X_t.astype(np.float64), y = y_t) # group label is not used
        
        print("\nMLP status check:") # Apply named_steps attribute of pipeline
        print("# of iterations (epochs): ", dnn.named_steps.mlp.n_iter_)
        print("Max allowable # of epochs: ", dnn.named_steps.mlp.max_iter)
        if dnn.named_steps.mlp.n_iter_ >= dnn.named_steps.mlp.max_iter:
            print("Warning: MLP is not converged!")
        print("\n# of layers: ", dnn.named_steps.mlp.n_layers_)
        print("# of hidden layers: ", dnn.named_steps.mlp.n_layers_ - 2)
        print("Activation function of hidden layers: ", \
              dnn.named_steps.mlp.activation)
        print("Activation function of output layer: ", \
              dnn.named_steps.mlp.out_activation_)
        print("Loss function type: ", dnn.named_steps.mlp.loss)
        print("Final loss: ", dnn.named_steps.mlp.loss_)
        
        # An n_feature by n_h1, n_h1 by n_h2, ... , n_hn by n_target list
        # print("\nCoefficients: ", dnn.named_steps.mlp.coefs_)
        # An 1 by n_h1, 1 by n_h2, ... 1 by n_hn, 1 by n_target list
        # print("\nIntercepts: ", dnn.named_steps.mlp.intercepts_)
        # print("\nLoss curve: ", dnn.named_steps.mlp.loss_curve_) # loss vs. epoch
        dnn_lcs.append(dnn.named_steps.mlp.loss_curve_)
        
        # Visualize DNN architecture:
        print("\nNow visualize DNN weights: ")
        VisualizeDNN(dnn.named_steps.mlp.coefs_, X_t.columns, [y_t.name])
        
        # Test set prediction via current fold fit model
        y_test_pred[:, cv_cur-1] = dnn.predict(X = X_test.astype(np.float64))
        # Multi-metric scoring:
        y_test_score[0, cv_cur-1] = skm.mean_absolute_error(y_test, \
                    y_test_pred[:, cv_cur-1])
        y_test_score[1, cv_cur-1] = math.sqrt(skm.mean_squared_error(y_test, \
                    y_test_pred[:, cv_cur-1]))
        y_test_score[2, cv_cur-1] = skm.r2_score(y_test, \
                    y_test_pred[:, cv_cur-1])
        print("\n---   ---")
        print("MAE: ", y_test_score[0, cv_cur-1])
        print("RMSE: ", y_test_score[1, cv_cur-1])
        print("RSQ: ", y_test_score[2, cv_cur-1])
        print("---   ---\n")       
        cv_cur += 1
    
    labels = np.arange(cv_folds)
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    for lc, cvid in zip(dnn_lcs, labels):
        LBL = 'CV fold # = ' + str(cvid+1)
        plt.plot(lc, label = LBL) # assign y only; x as index
    plt.legend(loc = 'best')
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    XlsxOutFName = 'myPSML_202005_4th_pred_case1.xlsx'
    """
    
    """
    # Case 2: Uncertainty qualification (2)
    # Stdev of predicted results from DNNs from different initialized weights
    ## Machine learning: (optimization and generalization check)
    RandStates = [0, 10, 50, 87, 100] # assign different random states
    nRS = len(RandStates)
    cv_cur = 1 # current cv fold
    cv_selected = 1
    train_i = []
    for train_ind, test_ind in gkf.split(X = X_train, y = y_train, \
                                               groups =  g_train):
        if cv_cur == cv_selected:
            train_i = train_ind # test_ind (validation set) is not used
        cv_cur += 1
    X_t = X_train.loc[X_train.index[train_i]]
    y_t = y_train.loc[y_train.index[train_i]]
    print("Check training+validation set size: ", X_train.shape[0])
    print("Check testing set size: ", X_test.shape[0])
    print("Check training set size only: ", X_t.shape[0])
    
    y_test_pred = np.zeros([X_test.shape[0], nRS])
    y_test_score = np.zeros([3, nRS])
    dnn_lcs = []
    for ii in range(0, nRS):
        # Set DNN with different random states:
        mlp = MLPRegressor(hidden_layer_sizes = HLNeurons, \
                           activation = AcFun, solver = Sol, alpha = Alpha, \
                           batch_size = MiniBS, max_iter = MaxIt, \
                           random_state = RandStates[ii], tol = TolErr, \
                           early_stopping = EStop, \
                           validation_fraction = ESVF, shuffle = Shuf, \
                           learning_rate = LearnRate, \
                           learning_rate_init = LRI, momentum = MTM, \
                           nesterovs_momentum = NestM)
        dnn = Pipeline([('scaler', prep), ('mlp', mlp)])
        dnn.fit(X = X_t.astype(np.float64), y = y_t) # group label is not used
        
        print("\nMLP status check for test run: ", ii+1)
        print("Random state = ", dnn.named_steps.mlp.random_state)
        print("# of iterations (epochs): ", dnn.named_steps.mlp.n_iter_)
        print("Max allowable # of epochs: ", dnn.named_steps.mlp.max_iter)
        if dnn.named_steps.mlp.n_iter_ >= dnn.named_steps.mlp.max_iter:
            print("Warning: MLP is not converged!")
        print("\n# of layers: ", dnn.named_steps.mlp.n_layers_)
        print("# of hidden layers: ", dnn.named_steps.mlp.n_layers_ - 2)
        print("Activation function of hidden layers: ", \
              dnn.named_steps.mlp.activation)
        print("Activation function of output layer: ", \
              dnn.named_steps.mlp.out_activation_)
        print("Loss function type: ", dnn.named_steps.mlp.loss)
        print("Final loss: ", dnn.named_steps.mlp.loss_)
        
        # An n_feature by n_h1, n_h1 by n_h2, ... , n_hn by n_target list
        # print("\nCoefficients: ", dnn.named_steps.mlp.coefs_)
        # An 1 by n_h1, 1 by n_h2, ... 1 by n_hn, 1 by n_target list
        # print("\nIntercepts: ", dnn.named_steps.mlp.intercepts_)
        # print("\nLoss curve: ", dnn.named_steps.mlp.loss_curve_) # loss vs. epoch
        dnn_lcs.append(dnn.named_steps.mlp.loss_curve_)
        
        # Visualize DNN architecture:
        print("\nNow visualize DNN weights: ")
        VisualizeDNN(dnn.named_steps.mlp.coefs_, X_t.columns, [y_t.name])
        
        # Test set prediction via current fold fit model
        y_test_pred[:, ii] = dnn.predict(X = X_test.astype(np.float64))
        # Multi-metric scoring:
        y_test_score[0, ii] = skm.mean_absolute_error(y_test,y_test_pred[:,ii])
        y_test_score[1, ii] = math.sqrt(skm.mean_squared_error(y_test, \
                    y_test_pred[:, ii]))
        y_test_score[2, ii] = skm.r2_score(y_test, y_test_pred[:, ii])
        print("\n---   ---")
        print("MAE: ", y_test_score[0, ii])
        print("RMSE: ", y_test_score[1, ii])
        print("RSQ: ", y_test_score[2, ii])
        print("---   ---\n")
    
    labels = np.arange(nRS)
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    for lc, runid in zip(dnn_lcs, labels):
        LBL = 'Test run # = ' + str(runid+1)
        plt.plot(lc, label = LBL) # assign y only; x as index
    plt.legend(loc = 'best')
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    XlsxOutFName = 'myPSML_202005_4th_pred_case2.xlsx'   
    """
    
    """
    ## Postprocessing (result analysis) of Case 1 and Case 2
    y_test_score_mean = np.mean(y_test_score, axis=1)
    y_test_score_std = np.std(y_test_score, axis=1)
    y_test_score_min = np.min(y_test_score, axis=1)
    y_test_score_max = np.max(y_test_score, axis=1)
    print("\n\nmyPSML_202005.main() now lists scores ...")
    print("Metric type \t mean \t std \t max \t min: ")
    print("MAE: ", y_test_score_mean[0], y_test_score_std[0], \
          y_test_score_max[0], y_test_score_min[0])
    print("RMSE: ", y_test_score_mean[1], y_test_score_std[1], \
          y_test_score_max[1], y_test_score_min[1])
    print("RSQ: ", y_test_score_mean[2], y_test_score_std[2], \
          y_test_score_max[2], y_test_score_min[2])
    
    y_test_pred_mean = np.mean(y_test_pred, axis=1)
    y_test_pred_std = np.std(y_test_pred, axis=1)
    y_test_pred_min = np.min(y_test_pred, axis=1)
    y_test_pred_max = np.max(y_test_pred, axis=1)
    # From numpy to pandas data structure:
    disp_pred = pd.Series(y_test_pred_mean, index = y_test.index, \
                          name = 'DNN-Mean')
    disp_pred_std = pd.Series(y_test_pred_std, index = y_test.index, \
                              name = 'DNN-Stdev')
    disp_pred_min = pd.Series(y_test_pred_min, index = y_test.index, \
                              name = 'DNN-Min')
    disp_pred_max = pd.Series(y_test_pred_max, index = y_test.index, \
                              name = 'DNN-Max')
    disp_pred_cvs = pd.DataFrame(y_test_pred, index = y_test.index, \
                                 columns = ['DNN']*cv_folds)
    # First row: lower errors; second row: upper errors
    err_minmax = pd.DataFrame([y_test_pred_mean - y_test_pred_min, \
                               y_test_pred_max - y_test_pred_mean])
    
    # Plotting:
    y_test_X = np.reshape(y_test.values, newshape = [y_test.shape[0], 1])
    reg = LinearRegression(fit_intercept = False)
    slope = reg.fit(X = y_test_X, y = disp_pred).coef_
    rsq = reg.score(X = y_test_X, y = disp_pred)
    XX = np.array([y_test.min(), y_test.max()])
    YY = slope*XX
    X0 = np.array([0.0, y_test.max()*1.25])
    STR = "Test set prediction with stdev, slope = " \
    + str(round(slope[0], 4)) + ", R^2 = " + str(round(rsq, 4))
    STR2 = "Test set prediction with min & max, slope = " \
    + str(round(slope[0], 4)) + ", R^2 = " + str(round(rsq, 4))
    
    print("\n\nmyPSML_202005.main() now plots predicted results ...")
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.errorbar(y_test, disp_pred, yerr = y_test_pred_std, c = 'b', \
                 marker = 'o', elinewidth = 1, ecolor = 'c', linewidth = 0, \
                 alpha = 0.5)
    plt.plot(XX, YY, color = 'r', linewidth = 2) # y_pred = slope*y_exp
    plt.plot(X0, X0, color = 'k', linewidth = 4) # X = Y line
    plt.xlabel("Expected disp. (um)")
    plt.ylabel("Predicted disp. (um)")
    plt.title(STR)
    plt.xlim(X0[0], X0[1])
    plt.ylim(X0[0], X0[1])
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    # standard deviation -> mean - min & max - mean
    plt.figure(figsize = (8, 6)) # default figsize = 6.4 x 4.8 (inches)
    plt.errorbar(y_test, disp_pred, yerr = err_minmax.values, c = 'b', \
                 marker = 'o', elinewidth = 1, ecolor = 'c', linewidth = 0, \
                 alpha = 0.5)
    plt.plot(XX, YY, color = 'r', linewidth = 2) # y_pred = slope*y_exp
    plt.plot(X0, X0, color = 'k', linewidth = 4) # X = Y line
    plt.xlabel("Expected disp. (um)")
    plt.ylabel("Predicted disp. (um)")
    plt.title(STR2)
    plt.xlim(X0[0], X0[1])
    plt.ylim(X0[0], X0[1])
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    
    # Export results to excel:
    print("\n\nmyPSML_202005.main() now exports predicted results ...")
    pred_df = pd.concat([g_test, X_test, y_test, disp_pred_cvs, disp_pred, \
                         disp_pred_std, disp_pred_max, disp_pred_min], axis = 1)
    pred_df.to_excel(XlsxOutFName, index = True)
    """
    
    # """
    ## Machine learning of Case 3 and Case 4: (optimization and prediction)
    # Case 3: Hyperparameter search (1)
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
    # """
    
    """
    # Case 4: Hyperparameter search (2)
    # Use grid search with narrower ranges of parameter set for fine tuning
    mlp = MLPRegressor(hidden_layer_sizes = HLNeurons, activation = AcFun, \
                       solver = Sol, alpha = Alpha, batch_size = MiniBS, \
                       max_iter = MaxIt, random_state = RandS, tol = TolErr, \
                       early_stopping = EStop, validation_fraction = ESVF, \
                       shuffle = Shuf, learning_rate = LearnRate, \
                       learning_rate_init = LRI, momentum = MTM, \
                       nesterovs_momentum = NestM)
    pipe_dnn = Pipeline([('scaler', prep), ('mlp', mlp)])
    
    dnn = GridSearchCV(estimator = pipe_dnn, cv = gkf, \
                       scoring = AssignedScore, refit = SetGridRefit, \
                       return_train_score = True, param_grid = param_gs)
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
    
    XlsxOutFName = 'myPSML_202005_4th_pred_case4.xlsx'
    """
    
    # """
    ## Postprocessing (model diagnosis) of Case 3 and Case 4
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
    # train_sizes1, train_scores1, test_scores1 = \
    # learning_curve(dnn.best_estimator_, X = Xohe.astype(np.float64), \
    #                y = yexp, groups = ylabel, scoring = DefaultMetric, \
    #                train_sizes = [1.0], cv = gkf_outer)
    
    # Check and plotting learning curve: (use all data set and test again)
    # Applied to find converged training data size and thus avoid underfitting
    # We found sklearn.learning_curve will reserve testing set, then dividing
    # the training and validation sets
    # ts_frac_ticks = np.linspace(0.2, 1.0, 5)
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
    # """   
    
    # """
    ## Postprocessing (result analysis) of Case 3 and Case 4
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
    # """
    
    ## Ending
    # Timer setting
    t_end = dt.datetime.now()
    print("\n\nmyPSML_202005.main() shows finish time: ", t_end)
    print("Total ellapsed time is: ", t_end - t_start)
    print("Finish myPSML_202005! Now exiting ...")
    print("=================================================")
# Done!!

