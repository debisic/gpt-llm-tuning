# Ubuntu Image
FROM ubuntu:20.04

# Install python dependencies
COPY requirements.txt .

# Install Python and pip
RUN apt-get update \
    && apt-get install -y python3.8 python3-pip \
    && pip3 install --upgrade pip \ 
    && pip3 install -r requirements.txt \
    && pip3 install fastapi[standard] uvicorn
# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

#Uvicorn
COPY ./app /usr/src/app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
