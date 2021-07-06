FROM python:3.6.8-slim

COPY src/* /src/
COPY requirements.txt /requirements.txt
COPY app.py /app.py
COPY download_model.sh /download_model.sh

RUN pip install -r requirements.txt
RUN bash download_model.sh

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app"]
