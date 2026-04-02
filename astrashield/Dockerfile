# AstraShield — "Predict. Prevent. Protect."
# National Space Hackathon 2026, IIT Delhi
# Base image: ubuntu:22.04 (required by grader)

FROM ubuntu:22.04

LABEL maintainer="AstraShield Team"
LABEL description="Autonomous Orbital Debris Intelligence & Defense System"
LABEL version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create runtime directories
RUN mkdir -p data logs

# Bootstrap: generate initial simulation state so first API call is instant
RUN python3 -c "
import sys, os
sys.path.insert(0, '.')
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
print('[AstraShield] Bootstrapping simulation state...')
from core.data_gen import generate_satellites, generate_debris, objects_to_dataframe
from core.clustering import run_clustering, compute_cluster_stats, tag_risk
sats = generate_satellites(50)
debs = generate_debris(1000)   # small bootstrap set for fast start
sat_df = objects_to_dataframe(sats)
deb_df = objects_to_dataframe(debs)
deb_df, _, _ = run_clustering(deb_df)
stats_df = compute_cluster_stats(deb_df)
deb_df = tag_risk(deb_df, stats_df)
sat_df.to_csv('data/satellites.csv', index=False)
deb_df.to_csv('data/debris.csv', index=False)
stats_df.to_csv('data/clusters.csv', index=False)
print('[AstraShield] Bootstrap complete — API ready.')
"

# Expose port (required by hackathon grader)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

# Bind to 0.0.0.0 (required — not localhost)
CMD ["python3", "api/server.py"]
