import streamlit as st
from transformers import pipeline
from pyngrok import ngrok
import requests


url = "None"
if not hasattr(st, 'already_started_server'):
    st.already_started_server = True

    st.write('''
            The first time this script executes it will run forever because it's
            running a FastAPI server.

            Just close this browser tab and open a new one to see your Streamlit
            app.
        ''')

    # from flask import Flask
    #
    # app = Flask(__name__)
    #
    # @app.route('/foo')
    # def serve_foo():
    #     return 'This page is served via Flask!'

    from typing import Optional
    from fastapi import FastAPI
    import os

    app = FastAPI()


    @app.get("/")
    def read_root():
        return {"Hello": "World"}


    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None):
        return {"item_id": item_id, "q": q}


    endpoint = ngrok.connect(8000).public_url
    print(' * Tunnel URL:', endpoint)
    status = requests.get(
        f"https://jarvis-ai-api.herokuapp.com/update_api_endpoint/?username=dipeshpal&token=5d57286c59a3c6d8c30e1d6675c0a6&endpoint={endpoint}")
    print("status: ", status)
    print("secrets: ",  st.secrets["token"])

    os.system('uvicorn main:app --reload')


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
