# Baseadas em Regras (Linguísticas)
# source venv/bin/activate
# pip install scikit-learn pandas streamlit
# streamlit run naive_bayes.py

import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Dados simples de exemplo
data = {
    'texto': [
        "Eu adorei o filme, foi excelente!",
        "O filme foi horrível, perdi meu tempo.",
        "Gostei muito, super recomendo.",
        "Péssimo. Atuação fraca e enredo ruim.",
        "Um dos melhores filmes que já vi!",
        "Que decepção. História sem graça.",
        "Fora Carille!",
    ],
    'sentimento': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 'negativo', 'negativo']
}

df = pd.DataFrame(data)

# Treinando o modelo
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['texto'], df['sentimento'])

# Função de predição para o Gradio
def predict_sentiment(text):
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = max(proba) * 100
    return prediction, f"{confidence:.2f}%"

# Interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Digite um texto aqui..."),
    outputs=[
        gr.Text(label="Sentimento previsto"),
        gr.Text(label="Confiança")
    ],
    title="Análise de Sentimento com Naive Bayes",
    description="Este app utiliza Naive Bayes com TF-IDF para prever o sentimento de textos em português."
)

iface.launch()
