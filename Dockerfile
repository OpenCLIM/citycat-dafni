FROM continuumio/miniconda3

RUN mkdir src

WORKDIR src

COPY run.py .
COPY environment.yml .

RUN conda env update -f environment.yml -n base

RUN dpkg --add-architecture i386
RUN apt update && apt-get install -y gnupg2 apt-transport-https ca-certificates
RUN wget -nc https://dl.winehq.org/wine-builds/winehq.key | apt-key add winehq.key
RUN echo "deb https://dl.winehq.org/wine-builds/debian/ stretch main" | tee -a /etc/apt/sources.list
RUN apt update
RUN apt install -y --install-recommends winehq-stable
COPY citycat.exe .

CMD [ "python", "run.py"]