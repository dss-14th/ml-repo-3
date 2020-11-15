
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class EDA_Graph():
    """
    With this class, you could explore the relation of each feature to 'voted' data.
    Input the path of data you want to analyze when you objectify this class.
    """
    
    def __init__(self, data):
        self.data = data
        self.df = self.read_data()
        self.columns = list(self.df.columns)
    
    # 1. dataset 불러오기
    def read_data(self):
        df = pd.read_csv(self.data)
        return df
    
    
    # 2. 각 Feature별 단순 투표량 비교 수치 시각화
    def nums_votes(self, col_name):
        '''
        `nums_votes` is a function drawing plot image that shows relation of each feature to numbers of 'voted' data.
        Input name of the column you want to explore in string type.
        '''
        
        if self.df[col_name].nunique() < 5:
            plt.figure(figsize=(8, 6))
            graph = sns.countplot(x=self.df[col_name], hue=self.df['voted'])
            
        elif self.df[col_name].nunique() >= 5:
            plt.figure(figsize=(12, 6))
            graph = sns.countplot(x=self.df[col_name], hue=self.df['voted'])
                
        else:
            plt.figure(figsize=(20, 6))
            graph = sns.countplot(x=self.df[col_name], hue=self.df['voted'])
            plt.legend(loc='upper center');
            
        return graph
    
    
    # 3. 각 Feature별 투표율 비교 수치 시각화
    def voting_rates(self, col_name):
        '''
        `voting_rates` is a function drawing plot image that shows relation of each feature to voting rates calculated from 'voted' data.
        Input name of the column you want to explore in string type.
        '''

        df_name = pd.crosstab(self.df[col_name], self.df['voted'])
        df_name["diff"] = (df_name["Yes"]/(df_name["Yes"]+df_name["No"]+df_name["0"])) 
        return df_name["diff"].sort_values().plot(kind="bar")

    
    # 4. 각 그래프 이미지 저장
    def save_as_img(self, *col_names, votes_num = True):
        '''
        `save_as_img` is a function saving plot images that show relation of each feature to 'voted' data.
        
        *col_names : string, one element or list
            Input name(s) of the column(s) you want to see the relation to 'voted' data.
        votes_num : bool, optional, 'True' is default
            Type 'True' if you want the graph with numbers of votes. Type 'False' if you want the graph with voting rates.
        ''' 
        
        ## graph 시각화 불가능 컬럼 제외
        col_names = list(col_names)
        if "major" in col_names:
            print("We cannot draw a graph of 'major' column.")
            col_names.remove("major")
            
        if "voted" in col_names:
            print("We cannot draw a graph of 'voted' column.")
            col_names.remove("voted")
            
        else:
            pass

        ## save plot img as png
        if votes_num == True:
            for col_name in col_names:
                self.nums_votes(col_name)
                plt.savefig('../MACH_data/graph_img/nums_votes_graph_{}.png'.format(col_name), dpi=200, bbox_inches="tight")
        else:
            for col_name in col_names:
                self.voting_rates(col_name)
                plt.savefig('../MACH_data/graph_img/voting_rates_graph_{}.png'.format(col_name), dpi=200, bbox_inches="tight")
