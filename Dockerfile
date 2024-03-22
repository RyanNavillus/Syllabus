# Use the official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/mylib


# Install any dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install tqdm

# You can also install additional tools you might need
RUN apt-get update && apt-get install -y --fix-missing\
    libgeos-dev \
    gcc \        
    g++ \        
    swig \       
    git \
    libgl1-mesa-dev \
    libglu1-mesa-dev \ 
    freeglut3-dev \ 
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Clone the multi_car_racing repository and install it
RUN git clone https://github.com/igilitschenski/multi_car_racing.git \
    && cd multi_car_racing \
    && pip install -e .

# Copy the Syllabus directory
COPY . .

WORKDIR /usr/src/mylib/syllabus/examples/experimental
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/mylib"

CMD ["/bin/bash"]

# docker build -t syllabus-image . 
# docker run -it --rm -p 4000:4000 syllabus-image
# docker run -it --rm -p 4000:4000 -v ${PWD}:/usr/src/mylib syllabus-image
# xvfb-run -s "-screen 0 1400x900x24" python -u multi_car_racing.py
# xvfb-run -s "-screen 0 1400x900x24" python -u multi_car_racing_v2.py