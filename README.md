# 데이콘 팀 

** 주의: 실행 시 constants.py에서 path를 colab인지 local인지에 따라 수정해줘야할 필요가 있음. 추후 일일히 변경할 필요 없게끔 수정 예정

1. 실행 파일

- augment_data.py : 데이터 증강
- prerprocess_data.py : 데이터 전처리 
- train.py : 학습
- test.py : 결과 분석

2. 유틸리티 모듈 (common 폴더)

- params.py : 실행에 필요한 모든 파라미터를 담고 있음
- eda.py : EDA에 필요한 모든 function들을 담고 있음
- engine.py :  모델 학습에 필요한 function
- load_data.py : 데이터를 불러오기
- utils.py : 기타등등

3. models

- /trained_models : 저장된 모델
- model_classes.py : 다양한 모델 정보를 담고 있ㅇ므

고유리, 이주현, 박기영, 이윤진, 박수현