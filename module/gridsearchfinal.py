import modeling_score as mdsc
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

class Modeling2(mdsc.Modeling):
    def __init__(self, *xy_train_test, lgrid):
        ada = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        self.grid_lgb= lgrid
        self.datas = []
        self.models = [ada, gbc, xgb, lgbm, self.grid_lgb]
        self.model_names = ['Ada', 'GBC', 'XGB', 'LGBM', "grid_lgb"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = xy_train_test
        
