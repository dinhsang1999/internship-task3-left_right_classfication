FROM python:3.6.8-slim

COPY src/* /src/
COPY requirements.txt /requirements.txt
COPY app.py /app.py
COPY download_model.py /download_model.py

RUN pip install -r requirements.txt
RUN python download_model.py

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app"]
