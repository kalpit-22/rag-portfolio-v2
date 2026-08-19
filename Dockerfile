# Use the official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies
RUN dnf update -y && dnf install -y \
    gcc \
    gcc-c++ \
    && dnf clean all

# Copy requirements and install them
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . ${LAMBDA_TASK_ROOT}

# Tell AWS Lambda to look for the Mangum handler we created in main.py
CMD ["main.handler"]
