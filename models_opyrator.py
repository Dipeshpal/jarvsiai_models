from pydantic import BaseModel, Field
from transformers import pipeline
import json
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

QNA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
CONVERSATIONAL_MODEL_NAME = "microsoft/DialoGPT-large"

qna_nlp = pipeline("question-answering", model=QNA_MODEL_NAME, tokenizer=QNA_MODEL_NAME)
# conversational = pipeline("conversational", model=CONVERSATIONAL_MODEL_NAME, tokenizer=CONVERSATIONAL_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(CONVERSATIONAL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(CONVERSATIONAL_MODEL_NAME)


class Input(BaseModel):
    task: str = Field(
        ...,
        description="Available Task: QnA, Conversational",
        example="QnA",
        max_length=50)
    qna_context: Optional[str] = None
    qna_question: Optional[str] = None
    qna_number_of_answers: Optional[int] = None
    conversational_chat_text: Optional[str] = None
    conversational_chat_history_ids: Optional[list] = None


class Output(BaseModel):
    results: Optional[str] = None
    chat_history_ids: Optional[list] = None


def ai_conversation(chat_text, chat_history_ids, step_new=3):
    # Let's chat for 3 lines
    for step in range(1):
        chat_history_ids = torch.IntTensor(chat_history_ids)
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(chat_text + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                  dim=-1) if step_new > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        print("bot_input_ids: ", bot_input_ids)
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        print("chat_history_ids: ", chat_history_ids)
        # pretty print last ouput tokens from bot
        ans = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        chat_history_ids = chat_history_ids.tolist()
        return ans, chat_history_ids


def ai_models_jarvis(input: Input) -> Output:
    """AI Models JarvisAI; Just an API call away"""

    if input.task == "QnA":
        print("input.qna_question: ", input.qna_question, "...", input.qna_context, "input.qna_number_of_answers: ", input.qna_number_of_answers)
        if input.qna_question is None or input.qna_context is None:
            return Output(
                results="'qna_question' and 'qna_context' can't be empty.")
        results = qna_nlp(
            {"question": input.qna_question, "context": input.qna_context},
            topk=input.qna_number_of_answers,
        )
        return Output(results=json.dumps(results))
    elif input.task == "Conversational":
        if input.conversational_chat_text is None:
            return Output(
                results="'conversational_chat_text' can't be empty. 'conversational_chat_history_ids' is optional.")
        if input.conversational_chat_history_ids is None:
            conversational_chat_history_ids_ = [[0]]
        else:
            conversational_chat_history_ids_ = input.conversational_chat_history_ids
        ans, chat_history_ids = ai_conversation(chat_text=input.conversational_chat_text, chat_history_ids=conversational_chat_history_ids_)
        return Output(results=ans, chat_history_ids=chat_history_ids)
    else:
        return Output(results="Help!!! Documentation coming soon...")
