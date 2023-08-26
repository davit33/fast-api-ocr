RUN apt-get update && apt-get install -y tesseract-ocr
COPY . /ocr
WORKDIR /ocr
EXPOSE 80