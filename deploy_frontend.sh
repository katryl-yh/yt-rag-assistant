#!/bin/bash
set -e

# Disable path conversion on Windows Git Bash
export MSYS_NO_PATHCONV=1

# Configuration - CHANGE THESE or set in environment
if [ -z "$ACR_NAME" ]; then
    echo "❌ Error: ACR_NAME environment variable is not set."
    echo "Please set it using: export ACR_NAME=your_registry_name"
    echo "Note: Registry name must be alphanumeric only (no hyphens)."
    exit 1
fi

# Validate ACR_NAME format (alphanumeric only)
if [[ ! "$ACR_NAME" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo "❌ Error: ACR_NAME '$ACR_NAME' contains invalid characters."
    echo "Registry names may contain only alpha numeric characters."
    exit 1
fi

IMAGE_NAME="yt-rag-frontend"
TAG="latest"

# Login to ACR
echo "🔐 Logging into ACR..."
az acr login --name $ACR_NAME

# Get the full registry URL
REGISTRY="${ACR_NAME}.azurecr.io"

# Set image tags
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "🏗️  Building Frontend image for AMD64..."
# Using --platform linux/amd64 is important for Azure Web App
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile \
  --tag $FULL_IMAGE_NAME \
  --push \
  .

echo "✅ Frontend image pushed: $FULL_IMAGE_NAME"

echo ""
echo "📦 Image built and pushed:"
echo "   - $FULL_IMAGE_NAME"
echo ""
echo "🚀 Next Steps:"
echo "1. Create an Azure Web App for Containers."
echo "2. Configure it to use the image: $FULL_IMAGE_NAME"
echo "3. Set the Environment Variable 'API_BASE_URL' in the Web App Configuration to point to your backend."
echo "✅ Build complete!"
