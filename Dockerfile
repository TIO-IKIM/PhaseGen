FROM ubuntu

USER root

COPY . .

RUN apt-get update && \
    apt-get install -y python3-pip python3

RUN pip install --no-cache-dir --break-system-package -r requirements.txt

CMD ["bash"]
