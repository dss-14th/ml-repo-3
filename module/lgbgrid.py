# gridsearch 클래스
import pickle
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

class GridSearch:    
    def __init__(self, *xy_train_test, **lgb_param):
        lgb = LGBMClassifier()
        
        lgb_param = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5, 10]
        }
        self.lgb_param = lgb_param
        self.gridsearch = GridSearchCV(estimator=lgb, param_grid=self.lgb_param, cv=5)
        self.X_train, self.X_test, self.y_train, self.y_test = xy_train_test
        
    def model_train(self):
        self.gridsearch.fit(self.X_train, self.y_train)
        
    def model_save(self):
        # Save model
        pickle.dump(self.gridsearch.best_estimator_, open("gridsearch.pickle", "wb"))
        
    def model_load(self):
        f = open("gridsearch.pickle", "rb")
        md = pickle.load(f)
        f.close()
        return md
