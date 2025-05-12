FROM python:3.10-slim-buster

# Accept GitHub credentials as build arguments
ARG GITHUB_NAME
ARG GITHUB_EMAIL

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    poppler-utils \
    tesseract-ocr \
    vim \
    curl \
    # libmagic-dev \
    # poppler-utils \
    # tesseract-ocr \
    # libgl1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh
    
# Set up Git to use HTTPS instead of SSH for GitHub
RUN git config --global url."https://github.com/".insteadOf git@github.com: && \
    git config --global url."https://".insteadOf git://

# Upgrade pip and install development tools
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pylint black isort

# Set the working directory in the container
WORKDIR /workspace

# Copy requirements.txt from the project root
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -ms /bin/bash vscode

# Set up Git config for the vscode user
USER vscode
RUN git config --global user.name "${GITHUB_NAME}" && \
    git config --global user.email "${GITHUB_EMAIL}"

# Set Python path
ENV PYTHONPATH "${PYTHONPATH}:/workspace"
