
class AddColumns:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def score(self, column="score", voted="voted",col_name="rate"):
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
