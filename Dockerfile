# Use the official AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies (required for some ML libraries like PyMuPDF or FAISS)
# Note: AWS Lambda base uses Amazon Linux (yum)
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    && yum clean all

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (mangum wrapper in main.py)
CMD ["main.handler"]
