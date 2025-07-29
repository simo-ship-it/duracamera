# ---------- Base ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

# ---------- Dipendenze di sistema ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv "$VIRTUAL_ENV" && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ---------- Librerie Python ----------
RUN pip install --no-cache-dir ultralytics opencv-python-headless \
                                 yt-dlp flask

# ---------- App ----------
WORKDIR /app
COPY durastream.py .

ENV VIDEO_URL="https://www.youtube.com/watch?v=Z49UkOi08DE"
EXPOSE 8000

ENTRYPOINT ["python", "durastream.py"]