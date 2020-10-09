FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc apt-utils

# RUN pip install --upgrade pip

WORKDIR /
COPY trained_model /trained_model
COPY . .

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
# RUN pip3 --no-cache-dir install tensorflow==2.9.1 aitextgen starlette uvicorn ujson
RUN pip3 --no-cache-dir install transformers==2.9.1 aitextgen starlette uvicorn ujson jinja2 aiofiles
COPY app.py /

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "app.py"]
