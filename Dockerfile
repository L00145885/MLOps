FROM python:3.10
WORKDIR /opt/flask_loan
COPY . /opt/flask_loan
RUN pip3 install -r requirements.txt
RUN python3.10 model.py
ENTRYPOINT FLASK_APP=/opt/flask_loan/flaskapp.py flask run --host=0.0.0.0
