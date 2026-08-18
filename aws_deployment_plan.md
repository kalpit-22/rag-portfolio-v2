# AWS Deployment Plan (Free Tier Architecture)

This document outlines the strategy for deploying the FastAPI/Docker RAG architecture to **Amazon Web Services (AWS)** at absolutely zero cost.

Since we have moved to a professional Dockerized FastAPI application, we can leverage the AWS Free Tier to host the API.

## Completely Free Deployment Options on AWS

Unlike Azure Container Apps which gives a monthly free grant but can quickly incur costs, AWS has options that are either free for 12 months, or completely free forever (Always Free Tier).

### Option 1: AWS EC2 (t2.micro / t3.micro) - *12 Months Free*
This involves spinning up a small Linux Virtual Machine and running your Docker container manually.
- **Pros:** Full control over the server. Very easy if you know basic Linux commands. 
- **Cons:** Only free for the first 12 months of your AWS account. You must manually manage the server and SSH into it.
- **Cost:** Free (750 hours/month = 24/7) for 12 months.

### Option 2: AWS Lambda + Function URL - *Always Free (Recommended for Zero Cost)*
AWS Lambda lets you run your code completely serverless. You only pay for exact execution time. By wrapping our FastAPI app with a small library called `Mangum`, we can deploy the entire app to a Lambda function!
- **Pros:** **Always free** tier includes **1 million free requests per month** forever. No server to manage or update. Scales automatically.
- **Cons:** Requires a minor code change to the FastAPI app (`pip install mangum`). "Cold starts" might cause a 1-3 second delay on the first request if the app hasn't been used in a while.
- **Cost:** $0, assuming you stay under 1 million requests and 400,000 GB-seconds of compute per month.

---

## The Workflow: EC2 Free Tier (Easiest Migration from Azure)

If you don't want to change any Python code, the simplest free option is an EC2 instance.

### 1. Build and Push the Image
Build the image for the target architecture (`linux/amd64` for standard EC2 instances).
```bash
docker build --platform=linux/amd64 -t your-dockerhub-username/portfolio-agent:latest .
docker push your-dockerhub-username/portfolio-agent:latest
```

### 2. Launch an EC2 Instance
1. Go to the AWS Console and search for **EC2**.
2. Click **Launch Instance**.
3. Select **Ubuntu 22.04 LTS** (Free tier eligible).
4. Select Instance Type: **t2.micro** or **t3.micro** (Free tier eligible).
5. Create a new Key Pair (save the `.pem` file to SSH in later).
6. Under Network Settings, allow **SSH (port 22)** and **Custom TCP (port 8000)** from Anywhere.
7. Click Launch.

### 3. Deploy the App
SSH into your instance using the `.pem` file you downloaded:
```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

Install Docker and run your container:
```bash
# Update and install Docker
sudo apt update -y && sudo apt install docker.io -y

# Run the container (pass all API keys here)
sudo docker run -d -p 8000:8000 \
    -e GOOGLE_API_KEY="your-key" \
    -e COHERE_API_KEY="your-key" \
    -e PINECONE_API_KEY="your-key" \
    -e LANGCHAIN_TRACING_V2="true" \
    -e LANGCHAIN_API_KEY="your-key" \
    -e LANGCHAIN_PROJECT="portfolio-rag" \
    your-dockerhub-username/portfolio-agent:latest
```

Your API is now live at `http://<your-ec2-public-ip>:8000` for free!

---

## The Workflow: AWS Lambda (Always Free)

If you want a truly "forever free" serverless solution that doesn't expire after 12 months, you can deploy your Docker container to AWS Lambda.

### 1. Modify the FastAPI App
To make FastAPI run on AWS Lambda, you need to add an adapter called `Mangum`.
In your code (`main.py` or similar):
```python
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

# Your existing routes...

# Add this at the very bottom:
handler = Mangum(app)
```

### 2. Build the Docker Image for AWS Lambda
AWS Lambda requires a slightly different base image and entrypoint. Update your `Dockerfile`:
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt mangum

# Copy your app code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["main.handler"]
```

### 3. Deploy to AWS Lambda
1. Create a private repository in **Amazon ECR** (Elastic Container Registry).
2. Build and push your Docker image to ECR.
3. Go to **AWS Lambda** -> **Create function**.
4. Select **Container image** and point it to your ECR image.
5. Once created, go to **Configuration -> Environment variables** and add your `GOOGLE_API_KEY`, etc.
6. Under **Configuration -> Function URL**, create a Function URL (Auth type: NONE) to get a free public HTTPS endpoint!

### 4. Automated CI/CD (GitHub Actions)
The repository is now pre-configured with a `.github/workflows/deploy.yml` file to automatically push to AWS Lambda on every commit to the `main` branch.

**Required GitHub Secrets for AWS:**
- `AWS_ACCESS_KEY_ID`: From your AWS IAM User.
- `AWS_SECRET_ACCESS_KEY`: From your AWS IAM User.

Make sure you update the `deploy.yml` placeholders (`portfolio-agent-repo` and `portfolio-agent-function`) with the exact names you used in Amazon ECR and AWS Lambda!

---

## Summary of Changes
1. **EC2 Route:** Requires manual SSH setup but uses your exact same Docker image. Free for 12 months.
2. **Lambda Route:** Requires `Mangum` and an AWS Lambda-specific Docker image but is 100% serverless and Always Free (1M requests/mo).
3. **CI/CD:** The `deploy.yml` workflow is fully updated to automate the Lambda ECR deployment.
