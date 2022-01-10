# Your starting image source
FROM python:3.8

# Working directory
WORKDIR /app

#Installing dependencies
RUN apt-get update && apt-get install libgomp1 -y 

# Install any Python dependecies we need
COPY requirements.txt .
RUN pip3 install -r requirements.txt


# COPY our /app files into the image
COPY /src .

# Command our Docker container will run to generate a submission CSV
CMD ["python3"]
