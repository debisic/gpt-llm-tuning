""" Implementing an API to test the model"""

from inference import *
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
def read_root():
    """ Welcome page of the API """
    return {'message': 'Welcome to the model API, to test the model go to IP of VM/docs'}

@app.post("/generate/")
def generate_text(prompt: str):
    """Get the response form our model"""
    return {'response': prediction(prompt) }
