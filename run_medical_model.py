import joblib

# بارگذاری مدل و بردار ویژگی‌ها
model = joblib.load('medical_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# تابع پرسش از مدل
def ask_model(question):
    question_vec = vectorizer.transform([question])
    answer = model.predict(question_vec)
    return answer[0]

# برنامه‌ی تعامل
print("مدل بارگذاری شد. سوال خود را بپرسید (برای خروج 'exit' تایپ کنید):")
while True:
    question = input("سوال: ")
    if question.lower() == 'exit':
        print("خروج از برنامه.")
        break
    answer = ask_model(question)
    print(f"جواب: {answer}")

