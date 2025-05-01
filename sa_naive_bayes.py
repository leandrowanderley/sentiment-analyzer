# Baseada em Machine Learning

import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

data = {
    'texto': [
        "Eu adorei o filme, foi excelente!",
        "O filme foi horrível, perdi meu tempo.",
        "Gostei muito, super recomendo.",
        "Péssimo. Atuação fraca e enredo ruim.",
        "Um dos melhores filmes que já vi!",
        "Que decepção. História sem graça.",
        "Fora Carille!",
        "Vamos!!! Carille foi demitido",
    ],
    'sentimento': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 'negativo', 'negativo', 'positivo']
}

df = pd.DataFrame(data)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['texto'], df['sentimento'])

def predict_sentiment(text):
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = max(proba) * 100
    return prediction, f"{confidence:.2f}%"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Digite um texto aqui..."),
    outputs=[
        gr.Text(label="Sentimento previsto"),
        gr.Text(label="Confiança")
    ],
    title="Analisador de Sentimentos",
    description="Insira um texto para ser analisado usando Naive Bayes."
)

iface.launch()
