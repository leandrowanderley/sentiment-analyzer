# Baseado em Deep Learning

# Positive: "I loved the news, looking forward to this" - "Feeling: Label_2 (Score: 0.99)"
# Neutral: "yeah, kinda liked, but i also didnt like it, but not that much, is ok" - "Feeling: Label_1 (Score: 0.47)"
# Negative: "I hated this video, looks horrible!!!!"- "Feeling: Label_0 (Score: 0.98)"

from transformers import pipeline
import gradio as gr

local_model_path = "./twitter-roberta-sentiment-local"
classifier = pipeline("sentiment-analysis", model=local_model_path, tokenizer=local_model_path)

def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    if label == "LABEL_0":
        label = "negative"
    elif label == "LABEL_1":
        label = "neutral"
    elif label == "LABEL_2":
        label = "positive"
    return f"{score:.2f}", label

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Type a text..."),
    outputs=[
        gr.Text(label="Reliability"),
        gr.Text(label="Feeling")
    ],
    title="Sentiment Analyzer",
    description="Enter a text to analyze its sentiment using roBERTa model (Deep Learning)"
)

if __name__ == "__main__":
    iface.launch()
