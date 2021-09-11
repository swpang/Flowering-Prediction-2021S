# Flowering-Prediction-2021S

Simple explanation of what each file does.

download_data.ipynb (created in Google Colab, requires slight changes in data path to run locally)
- 기상청 개방자료포털 openAPI 이용 자료 다운로드 코드

preprocessing.ipynb (created in Google Colab, requires slight changes in data path to run locally)
- 기상 자료 사용하여 각 모델의 입력 데이터 생성 (GDD, Chill Days 계산)

createdata_python.py
- preprocessing을 통해 계산된 기본 값들을 실제 입력 형식으로 만들어주는 파일

cleanfiles.py
- 입력데이터의 정리/preprocessing

ml_python.py
- where the bulk of ML happens.

graphing_python.py
- 말그대로 그래프 그리기

results_python.py
- ML 학습의 결과를 받아들여 개화시기 예측을 하는 코드
