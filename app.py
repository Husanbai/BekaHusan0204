import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Загрузка и объединение данных ---
@st.cache
def load_and_prepare_data():
    df_fake = pd.read_csv("fake.csv")
    df_true = pd.read_csv("true.csv")

    df_fake['label'] = 1
    df_true['label'] = 0

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.dropna(subset=['label', 'text'])

    df.to_csv("fake_or_true_news.csv", index=False)
    return df

# --- Обучение модели ---
@st.cache
def train_model():
    df = load_and_prepare_data()
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, vectorizer

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Распознавание фейковых новостей", page_icon="📰")
st.title("📰 Распознавание фейковых новостей")
st.write("Введите заголовок или текст новости, чтобы определить, фейк это или нет.")

user_input = st.text_area("✏️ Введите новость:")

# Загружаем модель при первом запуске
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.vectorizer = train_model()

# Кнопка проверки
if st.button("🔍 Проверить"):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст.")
    else:
        vect = st.session_state.vectorizer.transform([user_input])
        prediction = st.session_state.model.predict(vect)[0]
        proba = st.session_state.model.predict_proba(vect)[0][prediction]

        result = "🚫 ФЕЙК" if prediction == 1 else "✅ Настоящая новость"
        st.subheader(result)
        st.caption(f"Уверенность модели: **{proba:.2f}**")
