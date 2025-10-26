
# Toxic Comment Detection

Neural network project for detecting toxic comments in English, with context analysis.  
Includes LSTM-based model, prediction script, and interactive Streamlit web interface.

---

## Features

- Classifies comments into 6 categories:  
  **toxic, severe_toxic, obscene, threat, insult, identity_hate**
- Interactive **Streamlit interface** with visual feedback (✅ / ❌) and toxicity progress bar
- Demonstrates the effect of **context on model predictions**
- Simple **prediction script** for quick evaluation of comments

---

## Example

| Comment | toxic | severe_toxic | obscene | threat | insult | identity_hate | Result |
|---------|-------|--------------|--------|--------|--------|---------------|--------|
| Are you autistic? | 0.67 | 0.04 | 0.09 | 0.01 | 0.41 | 0.09 | ⚠️ Likely toxic |
| Are you autistic? I am asking to understand how to better communicate with you. | 0.43 | 0.01 | 0.03 | 0.00 | 0.16 | 0.03 | ✅ Neutral |

---

## Authors

- **Glushkov Nikita**  
- **Tarasov Nikita**  
- **Garifyanov Amir**  

**Contact:** glushkow.nikita7v@gmail.com

---

## Requirements

```bash
pip install -r requirements.txt
Dependencies:

torch

streamlit

numpy

pandas

How to run
Streamlit interface
bash
Копировать код
streamlit run app.py
Enter a comment in the text area

See probabilities for each toxicity category (✅ / ❌)

Check overall toxicity level via the progress bar

Prediction script
bash
Копировать код
python predict.py "Your comment here"
Returns probabilities for each category

Gives a summary if the comment is likely toxic

Notes
Model is trained on Jigsaw dataset for English comments

Context around words can significantly affect predictions

LSTM model does not understand intention; it detects patterns from training data

License
This project is for educational purposes.