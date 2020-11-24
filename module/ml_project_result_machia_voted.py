
import preprocessing1st as pre1
import preprocessing_nth as prenth
import modeling_score as mdsc

# 1st preprocessing
pre = pre1.Preprocessing1st("../MACH_data/data.csv")
xy = pre.preprocessing_model()


# Nth preprocessing
nth = prenth.PreprocessingNth()

# feature_selection
fs = nth.feature_selection(*xy)

# feature_addition
fa = nth.feature_addition(*fs)


# Modeling & Result
md = mdsc.Modeling(*fa)
md.print_score()
md.models_score_df()
