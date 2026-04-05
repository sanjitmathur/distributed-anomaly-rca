FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models on build
RUN python -c "\
from pipeline.preprocessing import load_data, preprocess; \
from pipeline.feature_engineering import FeatureEngineer; \
from models.train_models import train_all_models; \
df = load_data(); \
fe = FeatureEngineer(); \
df_eng = fe.transform(df.drop(columns=['Class'])); \
df_eng['Class'] = df['Class'].values; \
result = preprocess(df_eng); \
train_all_models(result['X_train'], result['X_test'], result['y_test']); \
print('Models trained successfully') \
"

EXPOSE 8501 8000

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.headless=true"]
