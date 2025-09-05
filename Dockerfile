FROM python:3.11

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ascii-image-converter binary
RUN wget -q https://github.com/TheZoraiz/ascii-image-converter/releases/download/v1.13.1/ascii-image-converter_1.13.1_linux_amd64.tar.gz \
    && tar -xzf ascii-image-converter_1.13.1_linux_amd64.tar.gz \
    && mv ascii-image-converter /usr/local/bin/ \
    && chmod +x /usr/local/bin/ascii-image-converter \
    && rm ascii-image-converter_1.13.1_linux_amd64.tar.gz

# Clear webdriver-manager cache to avoid THIRD_PARTY_NOTICES bug
RUN rm -rf /root/.wdm || true

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
