
import preprocessing1st as pre1
import preprocessing_nth as prenth
import modeling_score as mdsc
import lgbgrid as lgb_grid
import gridsearchfinal as final_grid
# 1st preprocessing
pre = pre1.Preprocessing1st("../MACH_data/data.csv")
xy = pre.preprocessing_model()


# Nth preprocessing
nth = prenth.PreprocessingNth()

# feature_selection
fs = nth.feature_selection(*xy)

# feature_addition
fa = nth.feature_addition(*fs)

#gridsearchcv_lgb


# Modeling & Result
print("{}".format("beforegridsearch"*10))
print("{}".format("="*20))
md = mdsc.Modeling(*fa)
md.print_score()
md.models_score_df()
print("{}".format("aftergridsearch"*20))
print("{}".format("="*20))
grid=lgb_grid.GridSearch(*fa)
lgb_grid_model = grid.model_load()
final = final_grid.Modeling2(*fa, lgrid=lgb_grid_model)
final.print_score()
final.models_score_df()
