GROUP=$(shell id --group --name)
GID=$(shell id --group)
USER=$(shell id --user --name)
UID=$(shell id --user)
TAG_NAME=$(USER)/acs-sandbox

image: .IMAGE
.IMAGE: Dockerfile .environment.yml
	- docker rmi -f $(TAG_NAME)
	docker build \
	    --build-arg group=$(GROUP) \
	    --build-arg gid=$(GID) \
	    --build-arg user=$(USER) \
	    --build-arg uid=$(UID) \
	    --tag $(TAG_NAME) \
	    .
	echo "$(TAG_NAME)" > $@

.environment.yml: ../environment.yml
	cp $< $@

push: .PUSH
.PUSH: .IMAGE
	@ echo "The image '${TAG_NAME}' is built for the Docker image repository '${DOCKER_REPOSITORY}'."
	@ echo "You can set the name of this repository with the environment variable DOCKER_REPOSITORY.\n"
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker tag $(TAG_NAME) $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker push $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker pull $(DOCKER_REPOSITORY)/$(TAG_NAME)
	echo "$(DOCKER_REPOSITORY)/$(TAG_NAME)" > $@

clean:
	- docker rmi -f $(TAG_NAME)
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	rm -f .IMAGE .PUSH .environment.yml

.PHONY: image push clean
