#!/bin/bash
set -e
USER=`id --user --name` # cluster user
NAME="acs"              # container name
RESOURCES="-c 8 -m 64g" # resources allocated by the container

# find the name of the image (with or without prefix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -f "${SCRIPT_DIR}/.PUSH" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.PUSH")"
elif [ -f "${SCRIPT_DIR}/.IMAGE" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.IMAGE")"
else
    echo "ERROR: Could not find any Docker image. Run 'make' or 'make push' first!"
    exit 1
fi

echo "Found the Docker image ${IMAGE}"

# configuration
case "$1" in
    -r|--resources)
        RESOURCES="$2"
        echo "INFO: Resource configuration is now '${RESOURCES}'"
        shift 2
        ;;
    -h|--help)
        echo "USAGE: $0 [--resources \"-c <n_cores> -m <memory>\"] [args...]"
        exit 0
        ;;
    *)
        docker create \
            --tty --interactive --rm \
            --volume /home/$USER:/mnt/home \
            --name "${USER}-${NAME}" \
            $RESOURCES \
            $IMAGE \
            "${@:2}" # pass additional arguments to the container entrypoint
        docker start "${USER}-${NAME}"
        docker attach "${USER}-${NAME}"
        ;;
esac
