#!/bin/bash
docker run -it \
    --name lane_detect \
    --user ros \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all   \
    --env NVIDIA_DRIVER_CAPABILITIES=all  \
    --env DISPLAY=${DISPLAY}  \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/docker_ws \
    --network host \
    --runtime nvidia \
    humble_opencv:v1

