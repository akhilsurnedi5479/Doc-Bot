FROM alpine:latest
RUN apk update
RUN apk add \
    build-base \
    freetds-dev \
    g++ \
    gcc \
    tar \
    gfortran \
    gnupg \
    libffi-dev \
    libpng-dev \
    libsasl \
    openblas-dev \
    openssl-dev
RUN apk add py-pip
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN apk add --no-cache python3-dev
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
