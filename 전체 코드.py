# 전체코드
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# 파일 경로 설정
train_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\open\train.csv"
test_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\open\test.csv"
sample_submission_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\open\sample_submission.csv"

# CSV 파일 읽기
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(sample_submission_file_path)

# 테스트 데이터의 ID를 저장
test_ids = test['ID']

# 데이터 타입 변환
train['Age'] = train['Age'].astype(float)
test['Age'] = test['Age'].astype(float)

# BloodPressure 열을 [0, 40, 80, 115] 구간으로 나누기
train['bloodpressure_cut'] = pd.cut(train['BloodPressure'], bins=[0, 40, 80, 115], labels=['A', 'B', 'C'], right=False)
test['bloodpressure_cut'] = pd.cut(test['BloodPressure'], bins=[0, 40, 80, 115], labels=['A', 'B', 'C'], right=False)

# BloodPressure 열을 5개의 동일한 빈도로 나누기
train['bloodpressure_cut'] = pd.qcut(train['BloodPressure'], q=5, labels=['A', 'B', 'C', 'D', 'E'])
test['bloodpressure_cut'] = pd.qcut(test['BloodPressure'], q=5, labels=['A', 'B', 'C', 'D', 'E'])

# Glucose 열을 [0, 110, 170] 구간으로 나누기
train['glucose_cut'] = pd.cut(train['Glucose'], bins=[0, 110, 170], labels=['normal', 'caution'], right=False)
test['glucose_cut'] = pd.cut(test['Glucose'], bins=[0, 110, 170], labels=['normal', 'caution'], right=False)

# LabelEncoder를 사용하여 glucose_cut 인코딩
le = LabelEncoder()

# train 데이터 인코딩
le.fit(train['glucose_cut'])
train['glucose_cut'] = le.transform(train['glucose_cut'])

# test 데이터 인코딩
# train 데이터에 없는 값이 test 데이터에 있을 경우 대비
for label in test['glucose_cut']:
    if label not in le.classes_:
        le.classes_ = np.append(le.classes_, label)

test['glucose_cut'] = le.transform(test['glucose_cut'])

# 특성과 레이블 분리
x_train = train.drop(columns=['ID', 'Outcome'])
y_train = train['Outcome']
x_test = test.drop(columns=['ID'])

# 수치형 데이터 선택
numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns

# 데이터 표준화 준비
scaler = StandardScaler()
scaler.fit(x_train[numeric_features])

# 데이터 표준화 적용
x_train_scaled = scaler.transform(x_train[numeric_features])
x_test_scaled = scaler.transform(x_test[numeric_features])

# 표준화된 데이터를 원래 데이터프레임에 다시 할당
for index, feature in enumerate(numeric_features):
    x_train[feature + '_scaled'] = x_train_scaled[:, index]
    x_test[feature + '_scaled'] = x_test_scaled[:, index]

# 새로운 특성을 사용하여 훈련 데이터와 검증 데이터로 분리
x_train_scaled = x_train[[col + '_scaled' for col in numeric_features]]
x_test_scaled = x_test[[col + '_scaled' for col in numeric_features]]
x_train_scaled, x_val_scaled, y_train, y_val = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성 및 훈련
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train_scaled, y_train)

# 검증 데이터로 예측
val_y_pred = model.predict(x_val_scaled)
accuracy = accuracy_score(y_val, val_y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# 테스트 데이터 예측
test_predictions = model.predict(x_test_scaled)

# sample_submission 형식에 맞게 결과 저장
submission_df = pd.DataFrame({'ID': test_ids, 'Outcome': test_predictions})

# 결과 저장
submission_file_path = r"C:\Users\mytoo\OneDrive\바탕 화면\open\submission.csv"
submission_df.to_csv(submission_file_path, index=False)
print(f'Submission file saved to: {submission_file_path}')