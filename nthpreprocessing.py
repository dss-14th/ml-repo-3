import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold ,GridSearchCV)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

# ada, gbc, xgb, lgbm 모델링 클래스 
class modeling:    
    def __init__(self,X_train, X_test, y_train, y_test):
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        self.datas = []
        self.models = [ada, gbc, xgb, lgbm]
        self.model_names = ['Ada', 'GBC', 'XGB', 'LGBM']
        
        # train, test 분리작업까지 마친 상태로 받아오기
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # 분류모델 평가지표 계산함수 (AUC, ACC를 우선순위로 사용함)
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
    
    # 평가지표와 confusion matrix 출력 함수 
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
            
# N차 전처리 클래스(feature 선별, robust scaling, weight 컬럼 추가)
class NthPreprocessing(modeling):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
    
    def nth_preprocesing(self, feature_selection=False, scale_robust=False, feature_addition=False):
        if feature_selection==True:
            return self.feature_selection()
        if scale_robust==True:
            return self.scale_robust()
        if feature_addition==True:
            return self.feature_addition()
            
    
    def feature_selection(self):
        datas = []
        for model in self.models:
            model.fit(self.X_train, self.y_train)
        
        # 모델 fit 이후 feature_importances_ 0인 컬럼을 선별하기 위함.
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
        
        # 4가지 모델에서의 feature_importances_가 모두 0인 컬럼들의 교집합 확인. 
        ada_gbc1 = list(set(ada_list1).intersection(gbc_list1))
        ada_gbc_xgb1 = list(set(ada_gbc1).intersection(xgb_list1))
        ada_gbc_xgb_lgbm1 = list(set(ada_gbc_xgb1).intersection(lgbm_list1))
        
        # 해당 컬럼들을 제외한 X데이터를 생성
        self.X_train = self.X_train.drop(ada_gbc_xgb_lgbm1, axis=1)
        self.X_test = self.X_test.drop(ada_gbc_xgb_lgbm1, axis=1)
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_robust(self):
        num_cols = []
        for col in self.X_train.columns:
            if ("E" in col) | ("age" in col) |("family" in col) |("elapse" in col):
                num_cols.append(col)
                
        rbscale = RobustScaler().fit(self.X_train[num_cols])
        self.X_train_ro[num_cols] = rbscale.transform(self.X_train[num_cols])
        self.X_test[num_cols] = rbscale.transform(self.X_test[num_cols])

        return self.X_train, self.X_test, self.y_train, self.y_test

       
    def feature_addition(self, column="score", voted="voted",col_name="rate"):
        df_tr = pd.concat([self.X_train, self.y_train], axis=1)
        df_te = self.X_test
        add_all_tr = df_tr[[voted, column]].groupby(column).count()
        add_yes_tr = df_tr[[voted, column]].groupby(column).sum()
        add_no_tr = add_all_tr - add_yes_tr
        df_add_tr = round((add_yes_tr - add_no_tr)/ add_all_tr, 4)
        df_add_tr = df_add_tr.rename(columns={voted:col_name})
        
        df1_tr = pd.merge(left=df_tr, right=df_add_tr, how="left", right_index=True, left_on=column)
        df1_te = pd.merge(left=df_te, right=df_add_tr, how="left", right_index=True, left_on=column)
        self.train_X=df1_tr.drop(voted, axis=1)
        self.train_Y=pd.DataFrame(df1_tr[voted])
        self.test_X=df1_te.fillna(0)
        self.test_Y=self.y_test
        
        return self.train_X, self.test_X, self.train_Y, self.test_Y
