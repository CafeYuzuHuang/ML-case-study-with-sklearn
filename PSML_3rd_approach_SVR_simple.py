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

import sklearn.preprocessing as skpp
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import sklearn.metrics as skm
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR # Epislon-supporting vector machines
from sklearn.svm import NuSVR # Nu-supporting vector machines
from sklearn.linear_model import LinearRegression

## Global variables
# (1) Constants
Tiny = 1e-4

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

# (4) Hyper-parameters for machine learning:
SVR_C = 1e0 # default = 1; C = 1/lambda where lambda is the L2 reg. strength
SVR_kernel = 'rbf' # default = 'rbf'
SVR_gamma = 'scale' # default = 'scale' (previous ver.: 'auto')
SVR_tol = Tiny
SVR_maxit = -1 # max possible value
SVR_epis = 0.05 # used in epsilon-SVR, default = 0.1
SVR_nu = 0.5 # used in nu-SVR, default = 0.5

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
    
    ## Preprocessing:
    # Define features and targets
    test_df0 = df.loc[df.index <= 85] # Group AHa, AHb, and AI are excluded
    print("Check matrix size of test dataframe: ", test_df0.shape)
    X_0, Y_0 = TableTransform(test_df0) # For test
    
    # Encoding for categorical features; scaling for numerics
    # encoding -> data splitting -> training set and testing set scaling
    
    # One-hot encoding for categorical features
    Xohe_0 = FeatureOHE(X_0)
    
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
    gkf = GroupKFold(n_splits = cv_folds)
    ylabel = Y_0[Tar[2]] # by test #
    yexp = Y_0[Tar[0]] # displacement at each t
    
    cv_cur = 1 # current cv fold
    for train_ind, test_ind in gkf_outer.split(X = Xohe, y = yexp, \
                                               groups =  ylabel):
        if cv_cur == cv_chosen: # replace indices
            train_index, test_index = train_ind, test_ind
        cv_cur += 1
        # Outputs of gkf_outer.split are:
        # A l x n array records indices of training sets of n-fold CV
        # A m x n array records indices of testing sets of n-fold CV
        # l ~ (1 - 1/n)*N, m ~ 1/n*N, where N is # of rows of data
    
    X_train = Xohe.loc[Xohe.index[train_index]]
    X_test = Xohe.loc[Xohe.index[test_index]]
    y_train = yexp.loc[yexp.index[train_index]]
    y_test = yexp.loc[yexp.index[test_index]]
    g_train = ylabel.loc[ylabel.index[train_index]] # for train-validation split
    g_test = ylabel.loc[ylabel.index[test_index]]
    
    ##### 3rd approach: check prediction uncertainty vs. CV training set #####
    ## Pipelining standard scaling with estimator:
    
    # Skip one-hot encoded columns during standard scaling
    PSList_all = Xohe.columns[Xohe.columns.str.contains(pat = 'Formulation_')]
    MOT_all = Xohe.columns[Xohe.columns.str.contains(pat = 'Mask_open_type_')]
    OHE_all = np.concatenate([PSList_all, MOT_all], axis = 0)
    Num_col = [item for item in Xohe.columns if item not in OHE_all]
    prep = make_column_transformer((skpp.StandardScaler(), Num_col), \
                                   remainder='passthrough')
    
    # Case 1
    svr = Pipeline([('scaler', prep), \
                    ('svr', SVR(kernel = SVR_kernel, gamma = SVR_gamma, \
                                C = SVR_C, tol = SVR_tol, \
                                max_iter = SVR_maxit, epsilon = SVR_epis))])
    
    # Case 2 (needs longer computation time)
    # However, parameter nu defines the upper bound of error and lower bound of
    # SV, thus pre-specify the # of SVs!
    # (Statistics and Computing 14: 199-222, 2004)
    """
    svr = Pipeline([('scaler',  prep), \
                    ('svr', NuSVR(kernel = SVR_kernel, gamma = SVR_gamma, \
                                  C = SVR_C, tol = SVR_tol, \
                                  max_iter = SVR_maxit, nu = SVR_nu))])
    """
    
    # Obtain fit model by each CV fold
    print("Check training+validation set size: ", X_train.shape[0])
    print("Check testing set size: ", X_test.shape[0])
    cv_cur = 1
    y_test_pred = np.zeros([X_test.shape[0], cv_folds])
    y_test_score = np.zeros([3, cv_folds])
    for train_ind, test_ind in gkf.split(X = X_train, y = y_train, \
                                               groups =  g_train):
        # test_ind (validation set) is not used
        X_t = X_train.loc[X_train.index[train_ind]]
        y_t = y_train.loc[y_train.index[train_ind]]
        print("CV fold # = ", cv_cur)
        print("Check training set size: ", len(train_ind))
        print("Check validation set size: ", len(test_ind))
        # g_t = g_train.loc[g_train.index[train_ind]]
        svr.fit(X = X_t.astype(np.float64), y = y_t) # group label is not used
        if svr.named_steps.svr.fit_status_ == 0:
            XXX = svr.named_steps.svr.support_vectors_.shape[0]
            print("SVR is correctedly fit! Check results:")
            print("# of support vectors: ", XXX)
            print("SV ratio (%): ", float(XXX)/float(X_train.shape[0])*100)
        else:
            print("Warning: SVR is not correctedly fit ...")
        # Test set prediction via current fold fit model
        y_test_pred[:, cv_cur-1] = svr.predict(X = X_test.astype(np.float64))
       
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
                          name = 'SVM-Mean')
    disp_pred_std = pd.Series(y_test_pred_std, index = y_test.index, \
                              name = 'SVM-Stdev')
    disp_pred_min = pd.Series(y_test_pred_min, index = y_test.index, \
                              name = 'SVM-Min')
    disp_pred_max = pd.Series(y_test_pred_max, index = y_test.index, \
                              name = 'SVM-Max')
    disp_pred_cvs = pd.DataFrame(y_test_pred, index = y_test.index, \
                                 columns = ['SVM-CV']*cv_folds)
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
    
    # Learning curve: Applied to find converged training data size and thus avoid underfitting
    ts_frac_ticks = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = \
    learning_curve(svr, X = Xohe.astype(np.float64), \
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
    
    # Export results to excel:
    print("\n\nmyPSML_202005.main() now exports predicted results ...")
    pred_df = pd.concat([g_test, X_test, y_test, disp_pred_cvs, disp_pred, \
                         disp_pred_std, disp_pred_max, disp_pred_min], axis = 1)
    pred_df.to_excel('myPSML_202005_3rd_results_pred.xlsx', index = True)
    ##### End of 3rd approach #####
    
    ## Ending
    # Timer setting
    t_end = dt.datetime.now()
    print("\n\nmyPSML_202005.main() shows finish time: ", t_end)
    print("Total ellapsed time is: ", t_end - t_start)
    
    print("Finish myPSML_202005! Now exiting ...")
    print("=================================================")
# Done!!

