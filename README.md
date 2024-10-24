<div align='center'>

 
<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=250&section=header&text=Rec%20N%20Roll&fontSize=80&animation=fadeIn&fontAlignY=38&desc=Lv2%20Project&descAlignY=51&descAlign=80"/>
</p>

# 🏠📊 LV.2 RecSys 프로젝트 : 수도권 아파트 전세가 예측 모델

</div>

## 🏆 대회 소개
| 특징 | 설명 |
|:---:| --- |
| 대회 주제 | 네이버 부스트캠프 AI-Tech 7기 RecSys level2 - Competitive Data Science|
| 대회 설명 | 아파트의 주거 특성, 금융 지표 등 다양한 데이터를 바탕으로 수도권 아파트의 전세가를 예측하는 AI 알고리즘 대회 |
| 데이터 구성 | `train.csv, test.csv, sample_submission.csv ,subwayInfo.csv , interestRate.csv, schoolInfo.csv, parkInfo.csv` 총 일곱 개의 CSV 파일 |
| 평가 지표 | Mean Absolute Error (MAE)로 실제 전세가와 예측 전세가 간의 오차 측정 |
---
## 💻 팀 구성 및 역할
| 박재욱 | 서재은 | 임태우 | 최태순 | 허진경 |
|:---:|:---:|:---:|:---:|:---:|
|[<img src="https://github.com/user-attachments/assets/0c4ff6eb-95b0-4ee4-883c-b10c1a42be14" width=130>](https://github.com/park-jaeuk)|[<img src="https://github.com/user-attachments/assets/b6cff4bf-79c8-4946-896a-666dd54c63c7" width=130>](https://github.com/JaeEunSeo)|[<img src="https://github.com/user-attachments/assets/f6572f19-901b-4aea-b1c4-16a62a111e8d" width=130>](https://github.com/Cyberger)|[<img src="https://github.com/user-attachments/assets/a10088ec-29b4-47aa-bf6a-53520b6106ce" width=130>](https://github.com/choitaesoon)|[<img src="https://github.com/user-attachments/assets/7ab5112f-ca4b-4e54-a005-406756262384" width=130>](https://github.com/jinnk0)|
|EDA, Feature Engineering|EDA, Feature Engineering|EDA, Feature Engineering|EDA, Feature Engineering|EDA, Feature Engineering|
---
## 🏠📊 프로젝트 개요
| 개요 | 설명 |
|:---:| --- |
| 주제 | 이 대회의 목적은 아파트 전세가 예측을 통해 부동산 시장의 정보 비대칭성을 해소하는 데 기여하는 것입니다. 2022년 기준 한국 가구의 51.9%가 아파트에 거주하며, 아파트는 한국에서 주거 문화의 중심이자 주요 자산 증식 수단입니다. 가계 자산의 70% 이상을 차지하는 중요한 자산인 아파트의 전세 시장은 매매 시장과 밀접하게 연관되어 있으며, 부동산 정책 수립과 시장 예측에 중요한 지표로 활용됩니다. |
| 목표 | 주어지는 아파트의 주거 특성, 금융 지표, 공공시설 정보 등의 데이터를 활용하여 전세가를 예측하는 AI 알고리즘을 개발하는 것이 대회의 주된 목표입니다. |
| 평가 지표 | **Mean Absolute Error (MAE)** - 실제 값과 예측값의 평균 절대 오차 지표 |
| 개발 환경 | `GPU` : Tesla V100 Server 4대, `IDE` : VSCode, Jupyter Notebook, Google Colab |
| 협업 환경 | `Notion`(진행 상황 공유), `Github`(코드 및 데이터 공유), `Slack` , `카카오톡`(실시간 소통) |

### 타임라인

<div align='center'>
📅 2024.10.01 ~ 2024.10.24 <br>
<img width="800" alt="스크린샷 2024-10-24 오후 2 56 59" src="https://github.com/user-attachments/assets/3d6b4a91-3586-4555-88bf-05fc3f6480e0"></img></div>
</div>
---

## 🕹️ 프로젝트 실행
### 디렉토리 구조

```
📦 level2-competitiveds-recsys-01
├── 📁code
│   ├── cnn_mlp.py
│   ├── tabtrasformer_main.py
│   ├── main.py
│   ├── 📁features
│   │   ├── README.md
│   │   ├── clustering_features.py
│   │   ├── count_features.py
│   │   ├── deposit_features.py
│   │   ├── distance_features.py
│   │   └── other_features.py
│   ├── 📁handler
│   │   ├── cnn_mlp_datasets.py
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   ├── 📁models
│   │   ├── 📁DL_tabtransformer
│   │   │   ├── dataset.py
│   │   │   ├── tabtransformer.py
│   │   │   └── trainer.py
│   │   ├── CombinedModel.py
│   │   ├── SeedEnsemble.py
│   │   ├── SpatialWeightMatrix.py
│   │   ├── XGBoostWithSpatialWeight.py
│   │   ├── inference.py
│   │   └── model.py
│   └── 📁utils
│       ├── common_utils.py
│       └── constant_utils.py
├── 📁data
│   ├── interestRate.csv
│   ├── parkInfo.csv
│   ├── sample_submission.csv
│   ├── schoolinfo.csv
│   ├── subwayInfo.csv
│   ├── test.csv
│   ├── train.csv
│   └── 📁transaction_data
│       ├── test_transaction_3.txt
│       ├── train_transaction_3.txt
│       └── valid_transaction_3.txt
└── 📁result
│   ├── mae
│   ├── submission
├── README.md
└── requirements.txt
```

### Installation with pip
1. `pip install -r requirements.txt` 실행
2. Unzip train, dev, test csv files at /data directory
3. Upload sample_submission.csv at /data directory
4. Execute `main.py` to run solution model
    
    Execute `cnn_mlp.py` to run CNN + MLP combined model
    
    Execute `tabtrasformer_main.py` to run TabTransformer model
