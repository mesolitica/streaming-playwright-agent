FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN playwright install-deps
RUN playwright install

RUN pip3 install Pillow

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY ./app /app

ENV PORT=9091
ENTRYPOINT ["/start-reload.sh"]