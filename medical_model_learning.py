import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. خواندن فایل CSV
data = pd.read_csv('medical_questions_answers.csv', header=None, names=['question', 'answer'])

# 2. جدا کردن سوالات و جواب‌ها
questions = data['question']
answers = data['answer']

# 3. تبدیل سوالات به بردار عددی (ویژگی‌ها)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# 4. تقسیم داده‌ها به آموزش و آزمایش
X_train, X_test, y_train, y_test = train_test_split(X, answers, test_size=0.2, random_state=42)

# 5. ساخت و آموزش مدل
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. ذخیره‌ی مدل و بردار ویژگی‌ها
joblib.dump(model, 'medical_model.pkl')  # ذخیره مدل
joblib.dump(vectorizer, 'vectorizer.pkl')  # ذخیره بردار ویژگی‌ها

print("مدل و بردار ویژگی‌ها ذخیره شدند.")

