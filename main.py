import streamlit as st
from transformers import pipeline
from pyngrok import ngrok
import requests


@st.cache(allow_output_mutation=True)
def load_model():
    question_answerer = pipeline("question-answering", model='bert-large-uncased-whole-word-masking-finetuned-squad')
    return question_answerer


def get_data(question, context):
    question_answerer = load_model()
    a = question_answerer(
        question=question,
        context=context
    )
    return a['answer']


url = "None"
if not hasattr(st, 'already_started_server'):
    st.already_started_server = True

    st.write('''
            The first time this script executes it will run forever because it's
            running a Opyrator server.

            Just close this browser tab and open a new one to see your Streamlit
            app.
        ''')

    import os

    endpoint = ngrok.connect(8000).public_url
    status = requests.get(
        f'https://jarvis-ai-api.herokuapp.com/update_api_endpoint/?username=dipeshpal&token={st.secrets["token"]}&endpoint={endpoint}')
    print("endpoint------------------------------", endpoint)
    # os.system("opyrator launch-ui op_asr:separate_audio --port 8000")
    os.system("uvicorn asr_fastapi:app --reload --reload-dir data")


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    st.title('Transformers Q and A Demo')

    # following lines create boxes in which user can enter data required to make prediction
    c = st.text_area("Context", "My name is Robert")
    q = st.text_input('Question', 'What is my name?')
    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = get_data(q, c)
        st.success(result)
        print(result)


if __name__ == '__main__':
    main()
