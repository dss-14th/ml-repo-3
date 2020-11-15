import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cates = {
    'education' : ['Less than high school','High school','University degree','Graduate degree'],
    'urban' : ['Rural','Suburban','Urban'],
    'gender' : ['Male','Female','Other'],
    'engnat' : ['Yes','No'],
    'hand' : ['Right','Left','Both'],
    'religion' : ['Agnostic','Atheist','Buddhist','Christian(Catholic)','Christian(Mormon)','Christian(Protestant)','Christian(othrer)','Hindu','Jewish','Muslim','Sikh','Other'],
    'orientation' : ['Heterosexual','Bisexual','Homosexual','Asexual','Other'],
    'race' : ['Asian','Arab','Black','Indigenous Australian','Native American','White','Other'],
    'voted' : ['1', '0'],
    'married' : ['Never married','Currently married','Previously married'],
}


class preprocessing1():
    
    def __init__(self, df):
        self.df = df
        
    def tran_cate(self, df, cate, x, y):
        self.df[cate] = self.df[cate].astype('str').replace(x, y)
        return self.df[cate]

    def pre1(self):
        # 직관적 EDA를 위해 컬럼명 수정
        self.df.rename(columns = {"Q1A" : "Q1_TP_notell_2u", "Q2A" : "Q2_TP_ppl_nd_dangun", "Q3A" : "Q3_TN_do_moral", "Q4A" : "Q4_VN_ppl_good", "Q5A" : "Q5_VP_ppl_bad", "Q6A" : "Q6_TN_hnsty_best", "Q7A" : "Q7_TN_lying_bad", "Q8A" : "Q8_VP_ppl_lazy", "Q9A" : "Q9_MN_humble_hnst", "Q10A" : "Q10_TN_hnstly_ask", "Q11A" : "Q11_VN_leader_clean", "Q12A" : "Q12_TP_trust_trouble", "Q13A" : "Q13_VP_ppl_criminal", "Q14A" : "Q14_VN_ppl_brave", "Q15A" : "Q15_TP_abu_good", "Q16A" : "Q16_TN_ppl_good", "Q17A" : "Q17_VN_ppl_notbad", "Q18A" : "Q18_VP_komsu_better", "Q19A" : "Q19_MP_anrocksa_ok", "Q20A" : "Q20_VP_money_good",
                     "Q1E" : "Q1E_notell_2u", "Q2E" : "Q2E_ppl_nd_dangun", "Q3E" : "Q3E_do_moral", "Q4E" : "Q4E_ppl_good", "Q5E" : "Q5E_ppl_bad", "Q6E" : "Q6E_hnsty_best", "Q7E" : "Q7E_lying_bad", "Q8E" : "Q8E_ppl_lazy", "Q9E" : "Q9E_humble_hnst", "Q10E" : "Q10E_hnstly_ask", "Q11E" : "Q11E_leader_clean", "Q12E" : "Q12E_trust_trouble", "Q13E" : "Q13E_ppl_criminal", "Q14E" : "Q14E_ppl_brave", "Q15E" : "Q15E_abu_good", "Q16E" : "Q16E_ppl_good", "Q17E" : "Q17E_ppl_notbad", "Q18E" : "Q18E_komsu_better", "Q19E" : "Q19E_anrocksa_ok", "Q20E" : "Q20E_money_good",
                     "TIPI1":"TYP_out", "TIPI2":"TYP_fight", "TIPI3":"TYP_depnd", "TIPI4":"TYP_anx", "TIPI5":"TYP_try", "TIPI6":"TYP_quiet", "TIPI7":"TYP_warm", "TIPI8":"TYP_disorg", "TIPI9":"TYP_calm", "TIPI10":"TYP_stable",
                     "VCL6" : "VCL6_F", "VCL9" : "VCL9_F", "VCL12" : "VCL12_F"
                    }, inplace=True)        
        
        # score 컬럼 추가
        col_list = list(self.df.columns)
        pos_col = []
        neg_col = []

        for col in col_list:
            if "P" in col and "Y" not in col:
                pos_col.append(col)
            if "N" in col:
                neg_col.append(col)

        self.df["score"] = self.df[pos_col].sum(axis=1) + self.df[neg_col].apply(lambda x: 6 -x).sum(axis=1)
        
        # V, T, M score 컬럼 추가 
        v_score = []
        t_score = []
        m_score = []

        for col in col_list:
            if "T" in col:
                t_score.append(col)
            if "M" in col:
                m_score.append(col)
            if "V" in col:
                v_score.append(col)
        self.df["v_score"] = self.df[v_score].sum(axis=1)
        self.df["t_score"] = self.df[t_score].sum(axis=1)
        self.df["m_score"] = self.df[m_score].sum(axis=1)
                
        # 텍스트 데이터로 변환, 시간 데이터 초단위로 환산
        vcl_col = []
        sec_col = []

        for col in col_list:
            if "VCL" in col:
                vcl_col.append(col)
            if "E" in col:
                sec_col.append(col)
        
        self.df[vcl_col] = self.df[vcl_col].applymap(lambda x: str(x).replace("1", "know") if x==1 
                                           else str(x).replace("0", "n_know"))
        self.df[sec_col] = self.df[sec_col].apply(lambda x: round(x*0.001))
        
        for x in list(cates.keys()):
            for idx, y in enumerate(cates[x]):
                if x == 'race':
                    self.tran_cate(self.df, x, "{}".format((idx+1)*10) ,y)
                else:    
                    self.tran_cate(self.df, x, "{}".format(idx+1) ,y)

        # major 컬럼 drop
        self.df.drop(columns = "major", inplace = True)
        col_list2 = list(self.df.columns)
        
        # null, 0 데이터 제거
        self.df.dropna(inplace=True)
        zero_idx = []
        for col in col_list2:
            zero_idx += list((self.df[self.df[col] == 0].index))
        zero_idx = list(set(zero_idx))
        self.df.drop(zero_idx, inplace=True)
        
        # 나이 18세 이하 데이터 drop
        self.df = self.df[self.df["age"]>17]
        
        # train, test로 나누기 
        df_X = self.df.drop('voted', axis=1)
        df_X = pd.get_dummies(df_X)
        
        df_y = self.df['voted'].astype('int')
        
                
        X_train, X_test, y_train, y_test=\
        train_test_split(df_X, df_y, test_size=0.2,
                         random_state=13, stratify=df_y)    

        
        return X_train, X_test, y_train, y_test
