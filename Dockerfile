FROM continuumio/miniconda3

RUN mkdir src

WORKDIR src

RUN dpkg --add-architecture i386 \
    && apt update && apt-get install -y gnupg2 apt-transport-https ca-certificates \
    && wget -nc https://dl.winehq.org/wine-builds/winehq.key | apt-key add winehq.key \
    && echo "deb https://dl.winehq.org/wine-builds/debian/ stretch main" | tee -a /etc/apt/sources.list \
    && apt update && apt install -y --install-recommends winehq-stable

COPY citycat.exe .

COPY environment.yml .
RUN conda env update -f environment.yml -n base

COPY run.py .

CMD [ "python", "run.py"]