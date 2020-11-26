<img src="https://user-images.githubusercontent.com/68367393/100315728-710eee00-2ffc-11eb-8dca-bfb0dd6ef589.png" width="100%" height="100%" alt="process">

# 머신러닝 분류모델 개발 프로젝트
[발표자료 다운로드(pdf)](https://github.com/dss-14th/ml-repo-3/raw/main/Who's_Voted.pdf)

**팀원**   
   박경원 [pkw-May](https://github.com/pkw-May)   
   전진경 [JJIN-1916](https://github.com/JJIN-1916)   
   정성용 [DODO-YONG](https://github.com/DODO-YONG)   


* * *
## 🔎 프로젝트 개요   

### 프로젝트 주제   
   개인의 [심리성향 테스트 답변]과 [인적사항 데이터]를 기반으로 [국가투표 참여여부] 예측하기   

### 활용 데이터   
- **데이터 내용**   
   마키아벨리즘 성향 테스트 답변 및 성격, 연령 등 인적사항 설문조사 답변 (총 73,490건)   

- **데이터 수집기간 & 방법**   
   2017.07~2019.03, 온라인 조사   

- **조사대상**   
   전세계 불특정 다수

- **데이터 출처**   
   심리학 공공데이터 사이트 (미국) <https://openpsychometrics.org/>

### 활용 모델   
- LGBM, GBC, XGB, ADA

### 모델링 프로세스   
<img src="https://user-images.githubusercontent.com/68367393/100344457-2f923900-3024-11eb-9626-1f3e736255a5.png" width="60%" height="60%" alt="process">

* * *
## 💻 핵심 프로세스   
### 1. EDA
\>> [edagraph.html](https://github.com/dss-14th/ml-repo-3/blob/main/EDA_data_association.html)   
\>> [edagraph.py](https://github.com/dss-14th/ml-repo-3/blob/main/edagraph.py)   
   <img src="https://user-images.githubusercontent.com/68367393/100344489-3caf2800-3024-11eb-9106-a80e1be9e03b.png" width="50%" height="50%" alt="1st_process">

### 2. 1차 전처리   
\>> [preprocessing1st.py](https://github.com/dss-14th/ml-repo-3/blob/main/module/preprocessing1st.py)   
   <img src="https://user-images.githubusercontent.com/68367393/100344538-4df83480-3024-11eb-8f42-ec3603e5b75b.png" width="50%" height="50%" alt="1st_process">

### 3. N차 데이터 전처리   
\>> [preprocessing_nth.py](https://github.com/dss-14th/ml-repo-3/blob/main/module/preprocessing_nth.py)   
   <img src="https://user-images.githubusercontent.com/68367393/100344573-5d777d80-3024-11eb-8602-139d8878ca91.png" width="50%" height="50%" alt="Nth_process">

### 4. 모델성능 고도화     
\>> [gridsearchfinal.py] (https://github.com/dss-14th/ml-repo-3/blob/main/module/gridsearchfinal.py)   
\>> [fianl_lgbm_model.pickle] (https://github.com/dss-14th/ml-repo-3/blob/main/module/lgb.pickle)
   <img src="https://user-images.githubusercontent.com/68367393/100344601-6d8f5d00-3024-11eb-9a8e-db1e0b021293.png" width="50%" height="50%" alt="Hyperparameter_Tuning">


* * *
## 📈 최종 모델 성능   
### AUC VALUE OF LGBM MODEL   
\>> [ml_project_result_machia_voted.py](https://github.com/dss-14th/ml-repo-3/blob/main/module/ml_project_result_machia_voted_final.py)   
   <img src="https://user-images.githubusercontent.com/68367393/100344639-7aac4c00-3024-11eb-850b-b5fe1ff79c6d.png" width="50%" height="50%" alt="Model_Performance">

* * *
## 📚 추가 해결 과제   
   <img src="https://user-images.githubusercontent.com/68367393/100344663-8566e100-3024-11eb-8352-4dbc39a1980d.png" width="50%" height="50%" alt="Review">

* * *
## 🧐 프로젝트 자세히 보기   
\>> [발표자료 다운로드(pdf)](https://github.com/dss-14th/ml-repo-3/raw/main/Who's_Voted.pdf)

* * *
## ⌨️ 사용언어 
   <img src="https://user-images.githubusercontent.com/68367393/99245317-b7b75800-2846-11eb-9b62-77be67f3b17b.png" width="7%" height="7%" alt="python">

