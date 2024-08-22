FROM nvcr.io/nvidia/pytorch:23.06-py3

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

# Update packages and install dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common git && \
    # CMake, Zlib development files, and build-essential
    apt-get install -y cmake zlib1g-dev build-essential && \
    # adds the DeadSnakes PPA, which provides newer Python versions
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # python 3.10, development headers, pip, and python virtual environment package
    apt-get install -y python3.10 python3.10-dev python3-pip python3.10-venv && \
    # use Python 3.10 as the default python command
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual env and path
ENV VIRTUAL_ENV=/syllabus
ENV PATH=/syllabus/bin:$PATH

# Set working directory
WORKDIR /home/app/syllabus

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install requirements
RUN pip install --quiet --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy all code to the container
COPY . .

RUN echo "Installing additional libraries ..."
RUN pip install multi-agent-ale-py griddly pettingzoo pufferlib \ 
    tensorboard pygame tqdm tyro colorama neptune python-dotenv && \
    pip install "autorom[accept-rom-license]" && \
    pip install jupyter ipykernel 
RUN AutoROM --accept-license

# expose the Jupyter notebook port
EXPOSE 8888

