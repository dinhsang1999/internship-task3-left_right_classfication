FROM python:3.6.8-slim

WORKDIR /app
ADD . /app/

RUN pip install -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app"]
