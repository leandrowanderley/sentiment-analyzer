# Baseadas em Regras Linguísticas

# Calcula 4 scores:
# pos → Positividade
# neu → Neutralidade
# neg → Negatividade
# compound → Score composto (de -1 a 1)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gradio as gr

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.25:
        feeling = "Positive"
    elif compound <= -0.25:
        feeling = "Negative"
    else:
        feeling = "Neutral"
    
    return scores, feeling

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Type a text here..."),
    outputs=[
        gr.JSON(label="Scoring"),
        gr.Text(label="Feeling")
    ],
    title="Sentiment Analyzer using VADER",
    description="This app analyzes sentiment using the VADER algorithm (lexicon & rule-based)."
)

iface.launch()
