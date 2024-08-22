# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif

# Set flag for docker run command
BASE_FLAGS=-it --rm --shm-size=1g -v ${PWD}:/home/app/syllabus -w /home/app/syllabus
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_RUN=docker run $(RUN_FLAGS) syllabus:latest
USE_CUDA = $(if $(GPUS),true,false)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) .

run:
	$(DOCKER_RUN) python $(example)

bash:
	$(DOCKER_RUN) bash

jupyter:
	$(DOCKER_RUN) jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root