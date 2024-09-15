FROM ubuntu:20.04

# Install python dependencies
COPY requirements.txt .

# Install Python and pip
RUN apt-get update \
    && apt-get install -y python3.8 python3-pip \
    && pip3 install --upgrade pip \ 
    && pip3 install -r requirements.txt 
    
# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set the default command
CMD ["/start.sh"]

# CMD ["python3", "main.py"]
