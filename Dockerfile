FROM python:3.9

WORKDIR /app/src

COPY . . 

RUN pip3 install -r requirements.txt

# WORKDIR /src

CMD ["uvicorn", "src.main:app",  "--reload", "--host", "0.0.0.0", "--port", "8000" ]