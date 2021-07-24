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
            running a FastAPI server.

            Just close this browser tab and open a new one to see your Streamlit
            app.
        ''')
    from typing import Optional
    from fastapi import FastAPI
    import os

    app = FastAPI()


    def load_model_local():
        question_answerer = pipeline("question-answering",
                                     model='bert-large-uncased-whole-word-masking-finetuned-squad')
        return question_answerer


    def get_data_local(question, context):
        question_answerer = load_model_local()
        a = question_answerer(
            question=question,
            context=context
        )
        return a['answer']

    @app.get("/")
    def read_root():
        ans = get_data_local("What is my name?", "My name is Dipesh")
        print(ans)
        return {"Hello": f"World: {ans}"}


    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None):
        return {"item_id": item_id, "q": q}


    endpoint = ngrok.connect(8888).public_url

    status = requests.get(
        f'https://jarvis-ai-api.herokuapp.com/update_api_endpoint/?username=dipeshpal&token={st.secrets["token"]}&endpoint={endpoint}')
    os.system('uvicorn main:app --host 127.0.0.1 --port 8888')


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
