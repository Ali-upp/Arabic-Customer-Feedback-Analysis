#!/usr/bin/env python3
"""Print model evaluation results in clean format"""

print("=" * 80)
print("نتائج تقييم نموذج تحليل رضا وشكاوى العملاء".center(80))
print("Customer Satisfaction & Complaints Analysis Model Evaluation".center(80))
print("=" * 80)
print()

from app.model_utils import load_model, load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data and model
df = load_data()
model = load_model()

print("📊 معلومات مجموعة البيانات (Dataset Information)")
print("-" * 80)
print(f"   إجمالي العينات (Total Samples): {len(df)}")
print(f"   عدد عينات الرضا (Satisfaction): {(df['label']=='0').sum()}")
print(f"   عدد عينات الشكاوى (Complaints): {(df['label']=='1').sum()}")
print()

# Split data same way as training
X = df['text'].astype(str)
y = df['label'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 نتائج الدقة (Accuracy Results)")
print("-" * 80)
print(f"   حجم مجموعة التدريب (Training Set Size): {len(X_train)} عينة")
print(f"   حجم مجموعة الاختبار (Test Set Size): {len(X_test)} عينة")
print()
print(f"   ✅ دقة النموذج (Model Accuracy): {accuracy * 100:.2f}%")
print()

print("📈 مصفوفة الالتباس (Confusion Matrix)")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred)
print(f"                      التنبؤ: رضا    التنبؤ: شكوى")
print(f"   الفعلي: رضا           {cm[0][0]:3d}            {cm[0][1]:3d}")
print(f"   الفعلي: شكوى         {cm[1][0]:3d}            {cm[1][1]:3d}")
print()

print("📝 تقرير التصنيف التفصيلي (Classification Report)")
print("-" * 80)
report = classification_report(y_test, y_pred, target_names=['رضا (0)', 'شكوى (1)'])
print(report)

print("=" * 80)
print("ملاحظات (Notes):".center(80))
print("=" * 80)
print("• تم تحسين النموذج بعد توسيع وموازنة مجموعة البيانات")
print("• Model improved after dataset expansion and balancing")
print("• تم استخدام TF-IDF Vectorizer + Logistic Regression")
print("• البيانات متوازنة بين الفئتين (Balanced dataset)")
print("=" * 80)
