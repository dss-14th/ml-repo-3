import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold ,GridSearchCV)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

class modeling:    
    def __init__(self,X_train, X_test, y_train, y_test):
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        self.datas = []
        self.models = [ada, gbc, xgb, lgbm]
        self.model_names = ['Ada', 'GBC', 'XGB', 'LGBM']
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_score(self, pred):
        acc = accuracy_score(self.y_test, pred)
        pre = precision_score(self.y_test, pred)
        rec = recall_score(self.y_test, pred)
        f1 = f1_score(self.y_test, pred)
        auc = roc_auc_score(self.y_test, pred)
       
        return acc, auc, pre, rec, f1
    
    def fit_model(self, model):

        model.fit(self.X_train, self.y_train)
        y_pre_tr = model.predict(self.X_train)
        self.y_pre_test = model.predict(self.X_test)
        total_score = self.get_score(self.y_pre_test)
        
        return total_score

    def models_score_df(self):
        cols_names = ['accuracy', 'AUC', 'precision', 'recall', 'f1']

        for model in self.models:
            self.datas.append(self.fit_model(model))

        return pd.DataFrame(self.datas, columns=cols_names, index=self.model_names)
    

    def print_score(self):
        datas = []
        for model in self.models:
            datas.append(self.fit_model(model))
        
            acc, auc, pre, rec, f1 = datas[0]
            con = confusion_matrix(self.y_test, self.y_pre_test)
            print('='*20)
            print(model)
            print('confusion matrix')
            print(con)
            print('='*20)

            print('Accuracy: {0:.4f}, AUC: {1:.4f}'.format(acc, auc))
            print('Recall: {0:.4f}, f1_score: {1:.4f}, precision: {2:.4f}'.format(rec, f1, pre))
            print('='*20)
            
            
class preprocessing_Nth(modeling):
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        super().__init__(X_train, X_test, y_train, y_test,**kwargs)
        
    def feature_selection(self, results=False):
        datas = []
        for model in self.models:
            model.fit(self.X_train, self.y_train)

        ada_fi = self.models[0].feature_importances_
        gbc_fi = self.models[1].feature_importances_
        xgb_fi = self.models[2].feature_importances_
        lgbm_fi = self.models[3].feature_importances_
        
        ada_fm = pd.DataFrame(zip(self.X_train.columns, ada_fi))
        ada_list1 = list(ada_fm[ada_fm[1]==0][0])
        gbc_fm = pd.DataFrame(zip(self.X_train.columns, gbc_fi))
        gbc_list1 = list(gbc_fm[gbc_fm[1]==0][0])
        xgb_fm = pd.DataFrame(zip(self.X_train.columns, xgb_fi))
        xgb_list1 = list(xgb_fm[xgb_fm[1]==0][0])
        lgbm_fm = pd.DataFrame(zip(self.X_train.columns, lgbm_fi))
        lgbm_list1 = list(lgbm_fm[lgbm_fm[1]==0][0])
        
        ada_gbc1 = list(set(ada_list1).intersection(gbc_list1))
        ada_gbc_xgb1 = list(set(ada_gbc1).intersection(xgb_list1))
        ada_gbc_xgb_lgbm1 = list(set(ada_gbc_xgb1).intersection(lgbm_list1))
    
        self.X_train = self.X_train.drop(ada_gbc_xgb_lgbm1, axis=1)
        self.X_test = self.X_test.drop(ada_gbc_xgb_lgbm1, axis=1)
        
        if results == True:
            result1 = self.models_score_df()
            return result1
        return self.X_train, self.X_test, self.y_train, self.y_test
