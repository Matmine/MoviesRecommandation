FROM apache/spark:3.4.0
 
# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
 
 
# Set the working directory
WORKDIR /app
 
# Copy the Spark project files to the container
COPY ./app /app
 
COPY ./requirements.txt /app
 
COPY ./app/ml-latest /ml-latest
 
# Install Python dependencies
RUN pip3 install -r requirements.txt
# RUN sed -i "s/localhost/$(curl http://checkip.amazonaws.com)/g" static/index.js
 
# Expose the port for the web application
EXPOSE 5432
 
# Set the entry point
CMD ["spark-submit", "server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]