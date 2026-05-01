FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless avoids needing system X11/GUI libs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py template.json ./

# Default: run the CLI; mount inputs at /data and outputs at /out.
ENTRYPOINT ["python3", "omr.py"]
CMD ["--help"]
