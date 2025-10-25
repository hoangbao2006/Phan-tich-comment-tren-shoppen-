# 💬 Project: Sentiment Analysis - Phân loại cảm xúc comment
# 🎓 Dành cho sinh viên năm 2 ngành AI

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Tạo dữ liệu comment thủ công
comments = [
    "Tôi rất thích sản phẩm này",
    "Quá tệ, tôi sẽ không mua lại",
    "Dịch vụ tốt và nhân viên thân thiện",
    "Hàng giao chậm và bị lỗi",
    "Giá rẻ mà chất lượng tuyệt vời",
    "Tôi thất vọng với trải nghiệm này",
    "Sản phẩm đáng tiền",
    "Chất lượng kém, không đúng mô tả",
    "Mọi thứ đều ổn, tôi hài lòng",
    "Không bao giờ quay lại nữa"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = tích cực, 0 = tiêu cực

df = pd.DataFrame({"Comment": comments, "Label": labels})
print("📊 Dữ liệu mẫu:")
print(df.head())

# 2️⃣ Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(df["Comment"], df["Label"], test_size=0.3, random_state=42)

# 3️⃣ Biến đổi text thành vector bằng Bag of Words
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4️⃣ Train model Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5️⃣ Dự đoán và đánh giá
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Độ chính xác: {acc*100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# 6️⃣ Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.title("Confusion Matrix - Sentiment Analysis")
plt.show()

# 7️⃣ Thử dự đoán comment mới
new_comments = [
    "Sản phẩm rất tốt, tôi yêu thích nó",
    "Tệ quá, không như mong đợi",
    "Nhân viên nhiệt tình và vui vẻ"
]

new_vec = vectorizer.transform(new_comments)
preds = model.predict(new_vec)

for c, p in zip(new_comments, preds):
    label = "Tích cực ✅" if p == 1 else "Tiêu cực ❌"
    print(f"💬 '{c}' → {label}")
