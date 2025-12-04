# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /recommendation_app

COPY recommendation_app/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python" ]

CMD ["./recommendation_app/wsgi.py"]