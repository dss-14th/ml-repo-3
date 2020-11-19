
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
class Modeling:    
    def __init__(self, *xy_train_test):
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        self.datas = []
        self.models = [ada, gbc, xgb, lgbm]
        self.model_names = ['Ada', 'GBC', 'XGB', 'LGBM']
        
        self.X_train, self.X_test, self.y_train, self.y_test = xy_train_test
        
        
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

        df = pd.DataFrame(self.datas, columns=cols_names, index=self.model_names)
            
        return print(df) 
    
    
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
