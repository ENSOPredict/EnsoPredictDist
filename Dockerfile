FROM ubuntu:latest

RUN apt-get update && apt-get -y update

RUN apt-get install -y build-essential python3.7 python3-pip python3-dev

RUN pip3 -q install pip --upgrade

RUN mkdir src

WORKDIR src/

COPY requirements.txt .

COPY nino34index_ml_prediction.py .

COPY nino34index_ml_prediction.ipynb .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8888

ENV TINI_VERSION v0.6.0

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini

RUN chmod +x /usr/bin/tini

#ENTRYPOINT ["/usr/bin/tini", "--"]

#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

#ENTRYPOINT "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" && /bin/bash
