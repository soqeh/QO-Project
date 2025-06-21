# --- Dockerfile for Hypercharge Reproducibility -----------------
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# install core dependencies
RUN pip install --no-cache-dir numpy scipy sympy pandas matplotlib pyyaml

# optional: set UTF-8 locale if needed
ENV PYTHONIOENCODING=UTF-8

# default command runs the pipeline
CMD ["python", "run_all.py"]