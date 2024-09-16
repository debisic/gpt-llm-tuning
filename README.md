# INTRODUCTION 

This project showcases expertise in leveraging generative AI models by optimizing and fine-tuning Large Language Models (LLMs) using pre-trained models from Hugging Face. Additionally, the project will be integrated into a CI/CD pipeline using GitHub Actions, ensuring continuous integration and high code quality standards.

To achieve these objectives, the [GPT-2 model](https://huggingface.co/openai-community/gpt2) was selected for an instruction tuning task. This task aims to enhance the model's ability to understand and follow user-provided instructions. The model will be fine tuned using the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca), which includes a diverse set of instruction-response pairs, to improve its performance across various tasks.

# SETUP INSTRUCTION

Setting up the project locally, first clone this repository using the command below, or download the zipped file
```
git clone https://github.com/debisic/gpt-llm-tuning.git
```
- 
Hardware requirements on Amazon EC2
OS - Linux/Ubuntu
AMI - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 
EBS - at least 60 GB, 
SSH and HTTP ports opened

# USER GUIDE

## Training the model
Once in the EC2 run the following command to activate pytorch env :
```
source activate pytorch 
```

Building Docker file for training
```
gpt-llm-tuning \
sudo docker build -f ./docker/train.Dockerfile -t gpt-llm ./docker
```

Run Docker with gpus and starting the training mapping the volume of the VM to the docker image
```
sudo docker run -it --gpus all -v /home/ubuntu/gpt-llm-tuning:/usr/src/app gpt-llm
```
## Running inferences with API

Building Docker file for inference
```
sudo docker build -f ./docker/api.Dockerfile -t server ./docker
```

Launching FastAPI server (Uvicorn)
```
docker run -it --gpus all -v /home/ubuntu/gpt-llm-tuning/docker/app:/usr/src/app -p 80:8000 server
```
In order to visualize the default UI for testing the model through the API, you need to get the public IP of the ec2 instance and map it to port ```80/docs```` and run it in the web browser
For example 10.26.224.133:80/docs where 10.26.224.133 is the public IP of the ec2 instance.



# IMPLEMENTATION EXPLANATION
1 - Model Selection

GPT-2 is an autoregressive language model based on the Transformer architecture. It has several key features that make it ideal for instruction tuning:

Transformer-Based: GPT-2 uses a multi-layer, attention-based Transformer architecture, which excels at capturing long-range dependencies and generating coherent, contextually aware text. This is particularly beneficial for instruction tuning, where the model must understand and generate meaningful responses based on the given prompt.

Pre-trained on Large-Scale Data: GPT-2 has been pre-trained on vast amounts of internet text data, making it highly versatile in generating human-like responses across a wide range of topics and contexts. This pre-training forms a strong foundation for fine-tuning on more specific tasks like instruction following.

Model Size(124M parameters) : The model's moderate size allows for relatively efficient fine-tuning using modern techniques like QLoRA (Quantized Low-Rank Adaptation) or PEFT (Parameter-Efficient Fine-Tuning), which further reduce the computational cost while maintaining high performance.

Suitability for Instruction Tuning: GPT-2 excels at generating coherent text that maintains context, which is essential for instruction-based tasks. Its autoregressive nature allows it to generate step-by-step responses that align well with the structure of instruction-response pairs found in the Alpaca dataset.

2 - Quantization

Quantization is applied to GPT-2 to reduce its model size and improve inference speed. This lowers memory and computational costs, making deployment more efficient on resource-limited devices. Despite using lower precision (e.g., 4-bit or 8-bit), quantization maintains acceptable performance, ensuring faster and cost-effective real-time applications, such as serving an API. for the implementation see the [main.py ](https://github.com/debisic/gpt-llm-tuning.git/master/main.pyi)

3 - Model Fine-tuning

The Alpaca dataset was chosen for instruction tuning because it provides diverse instruction-response pairs, covers a wide range of tasks, and is specifically designed for fine-tuning models like GPT-2 to enhance their instruction-following abilities. Most of the processing have already been done for this dataset.

5 - API Creation 

The API, built using FastAPI, provides an efficient interface for interacting with the fine-tuned GPT-2 model. It includes a GET endpoint for basic access and a POST endpoint for generating text based on user input

The test was implemented using the requests library in Python to validate the API functionality:

GET Request: Sends a request to the root endpoint (http://0.0.0.0:8000/) with a timeout parameter to prevent the program from hanging.
Error Handling: The test includes exception handling for timeout, connection errors, and other request-related exceptions to ensure robustness.

