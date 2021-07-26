import streamlit as st
from transformers import pipeline
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
            running a FastAPI server.

            Just close this browser tab and open a new one to see your Streamlit
            app.
        ''')

    import os
    from pyngrok import ngrok


    def fastapi_models():
        os.system("uvicorn asr_fastapi:app --reload --reload-dir data")


    def opyrator_models():
        os.system("opyrator launch-api models_opyrator:ai_models_jarvis --port 8080")


    endpoint = ngrok.connect(8080).public_url
    status = requests.get(
        f'https://jarvis-ai-api.herokuapp.com/update_api_endpoint/?username=dipeshpal&token={st.secrets["token"]}&endpoint={endpoint}')
    print("endpoint------------------------------", endpoint)
    opyrator_models()


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    st.title('Transformers Q and A Demo [OP]')

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
