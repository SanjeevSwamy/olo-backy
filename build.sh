#!/usr/bin/env bash
set -e

echo "🔧 Installing Chrome and dependencies..."

# Update package lists
apt-get update -y

# Install necessary dependencies for Chrome
apt-get install -y wget gnupg software-properties-common

# Add Google's signing key
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -

# Add Google Chrome repository
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list

# Update package lists again
apt-get update -y

# Install Google Chrome
apt-get install -y google-chrome-stable

# Verify Chrome installation
which google-chrome-stable || echo "Chrome installation failed!"

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo "✅ Build complete!"
