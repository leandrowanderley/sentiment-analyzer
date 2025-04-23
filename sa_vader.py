# Baseadas em Regras (Linguísticas)
# source venv/bin/activate
# pip install vaderSentiment
# streamlit run vader.py

# Calcula 4 scores:
# pos → Positividade
# neu → Neutralidade
# neg → Negatividade
# compound → Score composto (de -1 a 1)

# O serviço foi INCRÍVEL!!!

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

st.title("Sentiment Analyzer using Vader")

text = st.text_area("Type a text:")

if st.button("Analyze"):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    st.write("**Scoring:**", scores)
    
    if scores['compound'] >= 0.25:
        st.success("Feeling: Positive")
    elif scores['compound'] <= -0.25:
        st.error("Feeling: Negative")
    else:
        st.info("Feeling: Neutral")
