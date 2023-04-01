FROM python:3.10.9

# Set the working directory in the container
RUN mkdir build

WORKDIR /build

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi

# Copy the rest of the project files to the container
COPY . .

# Expose the port that the application will be running on
EXPOSE 8080

WORKDIR /build/app

# Run the application
CMD python -m uvicorn main:app --host 0.0.0.0 --port 80