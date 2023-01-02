# Pull official base image
FROM python:3.7-slim

# Install system dependencies
RUN apt-get clean && apt-get update -y

# Copy contents
COPY ./src ./src
COPY ./requirements.txt ./requirements.txt
COPY ./settings.yml ./settings.yml

# Install system dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

# Set work directory
WORKDIR ./src/pipeline

# Run pipelines for each model. 
# CMD ["python", "club_score_cutoff_pipeline.py"]
CMD ["python", "pipeline.py"]
