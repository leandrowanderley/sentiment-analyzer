# Baseadas em Regras Linguísticas

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gradio as gr

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound > 0.30:
        feeling = "Positive"
    elif compound < -0.30:
        feeling = "Negative"
    else:
        feeling = "Neutral"
    
    return scores, feeling

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Type a text here..."),
    outputs=[
        gr.JSON(label="Reliability"),
        gr.Text(label="Feeling")
    ],
    title="Sentiment Analyzer",
    description="Enter a text to analyze its sentiment using VADER (Linguistic Rules)."
)

if __name__ == "__main__":
    iface.launch()
