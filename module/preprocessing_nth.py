
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

# N차 전처리 클래스(feature 선별, robust scaling, weight 컬럼 추가)
class PreprocessingNth():
    def __init__(self):
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        self.models = [ada, gbc, xgb, lgbm]
        self.model_names = ['Ada', 'GBC', 'XGB', 'LGBM']
        
        
    def feature_selection(self, *xy_train_test):
        X_train, X_test, y_train, y_test = xy_train_test
        
        datas = []
        for model in self.models:
            model.fit(X_train, y_train)
        
        # 모델 fit 이후 feature_importances_ 0인 컬럼을 선별하기 위함.
        ada_fi = self.models[0].feature_importances_
        gbc_fi = self.models[1].feature_importances_
        xgb_fi = self.models[2].feature_importances_
        lgbm_fi = self.models[3].feature_importances_
        
        ada_fm = pd.DataFrame(zip(X_train.columns, ada_fi))
        ada_list1 = list(ada_fm[ada_fm[1]==0][0])
        gbc_fm = pd.DataFrame(zip(X_train.columns, gbc_fi))
        gbc_list1 = list(gbc_fm[gbc_fm[1]==0][0])
        xgb_fm = pd.DataFrame(zip(X_train.columns, xgb_fi))
        xgb_list1 = list(xgb_fm[xgb_fm[1]==0][0])
        lgbm_fm = pd.DataFrame(zip(X_train.columns, lgbm_fi))
        lgbm_list1 = list(lgbm_fm[lgbm_fm[1]==0][0])
        
        # 4가지 모델에서의 feature_importances_가 모두 0인 컬럼들의 교집합 확인. 
        ada_gbc1 = list(set(ada_list1).intersection(gbc_list1))
        ada_gbc_xgb1 = list(set(ada_gbc1).intersection(xgb_list1))
        ada_gbc_xgb_lgbm1 = list(set(ada_gbc_xgb1).intersection(lgbm_list1))
        
        # 해당 컬럼들을 제외한 X데이터를 생성
        X_train = X_train.drop(ada_gbc_xgb_lgbm1, axis=1)
        X_test = X_test.drop(ada_gbc_xgb_lgbm1, axis=1)
        
        # feature_selection 결과 데이터 저장
        self.feature_selection_xy = X_train, X_test, y_train, y_test
        
        return self.feature_selection_xy
    
    

    def scale_robust(self, *xy_train_test):
        X_train, X_test, y_train, y_test = xy_train_test
        
        num_cols = []
        for col in X_train.columns:
            if ("E" in col) | ("age" in col) |("family" in col) |("elapse" in col):
                num_cols.append(col)
                
        rbscale = RobustScaler().fit(X_train[num_cols])
        X_train[num_cols] = rbscale.transform(X_train[num_cols])
        X_test[num_cols] = rbscale.transform(X_test[num_cols])
        
        # scale_robust 결과 데이터 저장
        self.scale_robust_xy = X_train, X_test, y_train, y_test

        return self.scale_robust_xy

       
    def feature_addition(self, *xy_train_test, column="score", voted="voted",col_name="rate"):
        X_train, X_test, y_train, y_test = xy_train_test

        df_tr = pd.concat([X_train, y_train], axis=1)
        df_te = X_test
        add_all_tr = df_tr[[voted, column]].groupby(column).count()
        add_yes_tr = df_tr[[voted, column]].groupby(column).sum()
        add_no_tr = add_all_tr - add_yes_tr
        df_add_tr = round((add_yes_tr - add_no_tr)/ add_all_tr, 4)
        df_add_tr = df_add_tr.rename(columns={voted:col_name})
        
        df1_tr = pd.merge(left=df_tr, right=df_add_tr, how="left", right_index=True, left_on=column)
        df1_te = pd.merge(left=df_te, right=df_add_tr, how="left", right_index=True, left_on=column)
        X_train=df1_tr.drop(voted, axis=1)
        y_train=pd.DataFrame(df1_tr[voted])
        X_test=df1_te.fillna(0)
        
        # feature_addition 결과 데이터 저장
        self.feature_addition_xy = X_train, X_test, y_train, y_test

        return self.feature_addition_xy
