# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /opt/program

# Copy the model, vectorizer, inference script, and serve script
COPY model.pkl vectorizer.pkl inference.py serve ./

# Convert 'serve' script to Unix-style line endings in case it has Windows-style line endings
RUN apt-get update && apt-get install -y dos2unix && dos2unix serve

# Ensure the 'serve' script is executable
RUN chmod +x serve

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

# Expose the port that the Flask app will run on
EXPOSE 8080

# Set the entrypoint to run the 'serve' script
ENTRYPOINT ["./serve"]
