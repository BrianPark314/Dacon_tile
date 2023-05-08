# 데이콘 팀 

** 주의: 실행 시 constants.py에서 path를 colab인지 local인지에 따라 수정해줘야할 필요가 있음. 추후 일일히 변경할 필요 없게끔 수정 예정

1. 실행 파일

- _data_prep.py : 데이터 전처리 (colab 실행 시 runtime을 cpu로 설정할 것)
- _train.py : 학습
- _test.py : 결과 분석

2. 유틸리티 모듈

- constants.py : 실행에 필요한 모든 파라미터를 담고 있음
- customImageFolder.py : 데이터 로드에 필요한 커스텀 데이터셋 클래스를 담고 있음
- eda.py : EDA에 필요한 모든 function들을 담고 있음
- models.py : 여러가지 CV 모델을 담고 있음

고유리, 이주현, 박기영, 이윤진, 박수현