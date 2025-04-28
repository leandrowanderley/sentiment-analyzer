# Positive: "I loved the news, looking forward to this" - "Feeling: Label_2 (Score: 0.99)"
# Neutral: "yeah, kinda liked, but i also didnt like it, but not that much, is ok" - "Feeling: Label_1 (Score: 0.47)"
# Negative: "I hated this video, looks horrible!!!!"- "Feeling: Label_0 (Score: 0.98)"


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import gradio as gr

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# Função para analisar o texto
def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return f"Feeling: {label.capitalize()} (Score: {score:.2f})"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Type a text..."),
    outputs="text",
    title="Sentiment Analyzer (Positive / Neutral / Negative)",
    description="Enter a text to analyze its sentiment."
)

if __name__ == "__main__":
    iface.launch()
