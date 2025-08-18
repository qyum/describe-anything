FROM python:3.11.11 AS base               

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -v .
 

 

COPY . .
 
EXPOSE  8000