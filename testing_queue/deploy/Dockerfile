# Build only in the authorized cloud preparation phase. No default mutable base image.
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
RUN python -c 'import sys; assert sys.version_info[:3] == (3,13,15), "Exact Python3.13.15 base required"'
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-kaggle.txt /opt/queue-build/requirements-kaggle.txt
COPY check_kaggle_environment.py /opt/queue-build/check_kaggle_environment.py
RUN python -m pip install --no-cache-dir -r /opt/queue-build/requirements-kaggle.txt \
    && python -m pip check \
    && python /opt/queue-build/check_kaggle_environment.py --out /opt/queue-build/package-check.json \
    && python -m pip freeze > /opt/queue-build/installed-freeze.txt
# Runtime UID, capabilities, network, mounts and exact image digest are enforced by runner, not this default.
USER 65532:65532
WORKDIR /tmp
