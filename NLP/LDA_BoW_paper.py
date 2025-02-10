import PyPDF2
import nltk
import re
import sys
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel



## 0. PDF 읽기
def read_pdf(file_path):
    with open(file_path, "rb") as f:
        return "".join([p.extract_text() + "\n" for p in PyPDF2.PdfReader(f).pages])


## 1. 텍스트 전처리 (한 번만 실행)
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if
              token not in stop_words and not re.fullmatch(r'\d+', token) and len(token) > 1]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


## 2. BoW, TF-IDF 변환 함수 (한 번만 실행)
def extract_features(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])

    word_list = vectorizer.get_feature_names_out()
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    word_freq = sorted(zip(word_list, word_counts), key=lambda x: x[1], reverse=True)

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform([text])
    tfidf_word_list = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(X_tfidf.sum(axis=0)).flatten()
    tfidf_freq = sorted(zip(tfidf_word_list, tfidf_scores), key=lambda x: x[1], reverse=True)

    return word_freq, tfidf_freq


## 3. LDA 최적 토픽 개수 찾기
def find_optimal_topics(dictionary, corpus, texts, start=2, limit=10, step=1):
    scores = []
    best_num_topics = start
    best_coherence = 0

    print("\n🔄 [LDA 최적 토픽 개수 찾기 진행 중...]")
    for num in range(start, limit, step):
        sys.stdout.write(f"\r▶ Checking num_topics = {num}... \n")
        sys.stdout.flush()

        lda_model = LdaModel(corpus, num_topics=num, id2word=dictionary, passes=30, iterations=150)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        scores.append((num, coherence_score))

        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_num_topics = num

    print("\n✅ 최적의 토픽 개수 결정 완료.")
    return best_num_topics


# ✅ 실행 코드 (출력 한 번만!)
if __name__ == "__main__":
    text = read_pdf("Data/Paper_PDF2_PointRCNNpdf.pdf")         # pdf 파일 선택

    # 전처리 (한 번만 실행)
    processed_text = preprocess_text(text)

    print("\n📌 [텍스트 전처리 결과]")
    print(f"전처리된 텍스트 일부 (앞 1000자):\n{processed_text[:1000]}")

    # BoW, TF-IDF (한 번만 실행)
    word_freq, tfidf_freq = extract_features(processed_text)

    print("\n📌 [BoW 변환 결과]")
    print(f"총 단어 개수 (Vocabulary Size): {len(word_freq)}")
    print(f"일부 단어 목록 (Top 20): {', '.join([word for word, _ in word_freq[:20]])}")

    print(f"\n📌 [빈도 높은 단어 Top 20]")
    for word, freq in word_freq[:20]:
        print(f"{word}: {freq}")

    print(f"\n📌 [TF-IDF 가중치 Top 20]")
    for word, score in tfidf_freq[:20]:
        print(f"{word}: {score:.4f}")

    # ✅ LDA 토픽 모델링 (여기서부터 반복 출력 안 됨)
    tokenized_text = processed_text.split()
    dictionary = corpora.Dictionary([tokenized_text])
    corpus = [dictionary.doc2bow(tokenized_text)]

    optimal_num_topics = find_optimal_topics(dictionary, corpus, [tokenized_text], start=2, limit=10)
    print(f"✅ 최적의 토픽 개수: {optimal_num_topics}")

    # ✅ **LDA 모델 최적의 num_topics로 학습**
    print("\n🔄 [LDA 모델 최종 학습 진행...]")
    lda_model = LdaModel(corpus, num_topics=optimal_num_topics,
                         id2word=dictionary, passes=50, iterations=300,
                         alpha='auto', eta='auto', random_state=42)

    # ✅ **최종 LDA 결과 출력**
    print(f"\n📌 [LDA 토픽 모델링 결과 (Topic number = {optimal_num_topics})]")
    for idx, topic in lda_model.show_topics(num_topics=optimal_num_topics, num_words=10, formatted=False):
        words_with_probs = " + ".join([f"{prob:.3f}*\"{word}\"" for word, prob in topic])  # 확률 포함
        words_only = ", ".join([word for word, _ in topic])  # 확률 제거한 단어 리스트

        print(f"- 토픽 {idx + 1}: {words_with_probs}")
        print(f"   => {words_only}\n")
