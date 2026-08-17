# Azure Deployment Plan (New Architecture)

This document outlines the deployment strategy for the upgraded FastAPI/Docker RAG architecture to **Microsoft Azure**. 

Since we have moved away from Streamlit to a professional Dockerized FastAPI application, Azure gives us some fantastic, highly scalable hosting options.

## Deployment Options on Azure

### Option 1: Azure App Service (Web App for Containers) - *Recommended for Simplicity*
This is the equivalent of a managed EC2 instance but specifically designed for Docker containers.
- **Pros:** Extremely easy to set up, built-in SSL/HTTPS, easy environment variable management in the Azure Portal.
- **Cons:** Free tier (F1) has limited RAM and compute minutes. Basic tier (B1) costs around $13/month.

### Option 2: Azure Container Apps (ACA) - *Recommended for Cost & Scaling*
This is Azure's serverless container offering (similar to AWS App Runner or Fargate).
- **Pros:** Scales to zero when no one is using it, meaning you only pay for exact execution time. Very generous free grant (2 million requests and 180,000 vCPU seconds free per month).
- **Cons:** Slightly more complex initial setup (requires setting up a Log Analytics workspace and Container Apps Environment).

---

## The Workflow (The Professional Way)

Just like we discussed with AWS, the best practice is to build the image locally (or in GitHub Actions) and push it to a registry, rather than building it on the Azure server.

### 1. Build the Image for AMD64
Even if you switch to your PC, it's a good habit to ensure you are building for the correct cloud architecture (Azure heavily uses `linux/amd64`).
```bash
docker build --platform=linux/amd64 -t your-dockerhub-username/portfolio-agent:latest .
```

### 2. Push to a Registry
You can use **Docker Hub** (free and easiest) or **Azure Container Registry (ACR)** (keeps everything inside Azure's ecosystem).
```bash
docker push your-dockerhub-username/portfolio-agent:latest
```

### 3. Deploy to Azure Container Apps (Example CLI)
Assuming you use Azure CLI (`az`), here is how you deploy it to a serverless Container App:

```bash
# 1. Create a Resource Group
az group create --name PortfolioResourceGroup --location eastus

# 2. Create a Container Apps Environment
az containerapp env create \
  --name portfolio-env \
  --resource-group PortfolioResourceGroup \
  --location eastus

# 3. Deploy the App and pass your Secrets!
az containerapp create \
  --name portfolio-agent-app \
  --resource-group PortfolioResourceGroup \
  --environment portfolio-env \
  --image your-dockerhub-username/portfolio-agent:latest \
  --target-port 8000 \
  --ingress 'external' \
  --env-vars \
    GOOGLE_API_KEY="your-key" \
    COHERE_API_KEY="your-key" \
    PINECONE_API_KEY="your-key" \
    LANGCHAIN_TRACING_V2="true" \
    LANGCHAIN_API_KEY="your-key" \
    LANGCHAIN_PROJECT="portfolio-rag"
```
Azure will instantly provide you with an `https://...` URL where your app is hosted!

---

## Updating GitHub Actions for Azure (CI/CD)

If you want GitHub Actions to automatically deploy to Azure every time you push, you will update your `.github/workflows/deploy.yml`.

Instead of SSHing into an EC2 instance, you will use the `azure/login` action and the `azure/container-apps-deploy-action`.

**Required GitHub Secrets for Azure:**
1. `AZURE_CREDENTIALS`: A Service Principal JSON output used to authenticate GitHub with Azure.
2. `DOCKER_USERNAME` & `DOCKER_PASSWORD`.

**Workflow Example:**
```yaml
      - name: Log into Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
          
      - name: Deploy to Azure Container Apps
        uses: azure/container-apps-deploy-action@v2
        with:
          resourceGroup: PortfolioResourceGroup
          containerAppName: portfolio-agent-app
          imageToDeploy: ${{ secrets.DOCKER_USERNAME }}/portfolio-agent:latest
```

---

## Summary of Changes from the Old Architecture
1. **Ports:** Ensure Azure routes traffic to port **8000** (where FastAPI runs), not 8501 (where Streamlit ran).
2. **Observability:** Don't forget to inject the `LANGCHAIN_*` variables into your Azure environment configuration so LangSmith continues to trace your deployed queries.
3. **No Heavy Builds:** Azure will not waste time building the heavy Python AI libraries; it will just pull your sleek, pre-built Docker image instantly.
