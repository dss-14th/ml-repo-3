# MACHINE LEARNING MODELING PROJECT
[[PPT 다운로드_Download Presentation File]](https://github.com/dss-14th/ml-repo-3/raw/main/Who's_Voted.pdf)
* * *
## 🧐 이 사람, 투표했을까?
## Guess, Did He/She Vote?

- 머신러닝 분류모델 개발 프로젝트
   - Developing Machine Learning Classifier Model Project

   
- 팀원 : 박경원, 전진경, 정성용     
   - contributors : pkw-May[https://github.com/pkw-May], JJIN-1916[https://github.com/JJIN-1916], DODO-YONG[https://github.com/DODO-YONG]    

* * *
## 🔎 프로젝트 개요
## SUMMARY

### GOAL) 목표
- 개인의 심리성향 테스트 답변과 인적사항 데이터를 기반으로 국가투표 참여여부 예측   
   - Predicting the Voter using Machine Learning Classifier Model with his/her Personal Data.   

### DATA INFO) 활용 데이터
- 데이터 내용: 마키아벨리즘 성향 테스트 답변 및 성격, 연령 등 인적사항 설문조사 답변   
   - Contents of dataset : Individual answer datas given after MACHI-IV test and personal information survey.   

- 데이터 수집기간 & 방법: 2017.07~2019.03, 온라인   
   - Collecting Dates & Method: July 2017 - March 2019, via online   

- 데이터 출처: 심리학 공공데이터 사이트 (미국)
   - Source of dataset: <https://openpsychometrics.org/>

### USED MODEL) 활용 모델
- LGBM, GBC, XGB, ADA

### MODELING PROCESS) 모델링 프로세스
![image](https://user-images.githubusercontent.com/67700119/98928715-77da3300-251d-11eb-8523-b1a494bbd789.png)
- 1) 1차 데이터 전처리 (0, null 제거 등)
   - Data Cleaning (remove 0, null data ect...)
- 2) N차 데이터 전처리 (3단계)
   특성 데이터 선별
   - Feature Selection by Feature Importance value
- 3) Robust Scaling

### MODEL PERFORMANCE) 모델 성능
* * *
### [1차 전처리]
![image](https://user-images.githubusercontent.com/67700119/98930097-7dd11380-251f-11eb-920a-c4e9bb732481.png)

### [N차 전처리]
![image](https://user-images.githubusercontent.com/67700119/98930138-8e818980-251f-11eb-9698-ca2a0e4a0b06.png)

## 결과
try https://github.com/dss-14th/ml-repo-3/wiki/merong
## 한계점 

## 팀 구성원 & 역할

## 사용언어 
