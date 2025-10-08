#!/bin/bash
# copy_code.sh - copy local code into container
# TODO: git branch 만들기

docker cp vlm_manipulation/main.py metasim_2:/home/lukesong_google_com/RoboVerse/vlm_manipulation/ && \
docker cp vlm_manipulation/curobo_utils.py metasim_2:/home/lukesong_google_com/RoboVerse/vlm_manipulation/ && \
docker cp vlm_manipulation/gsnet_utils.py metasim_2:/home/lukesong_google_com/RoboVerse/vlm_manipulation/ && \
docker cp vlm_manipulation/metasim_utils.py metasim_2:/home/lukesong_google_com/RoboVerse/vlm_manipulation/ && \
docker cp metasim/sim/isaaclab/isaaclab.py metasim_2:/home/lukesong_google_com/RoboVerse/metasim/sim/isaaclab/ && \
docker cp metasim/utils/kinematics_utils.py metasim_2:/home/lukesong_google_com/RoboVerse/metasim/utils/ && \
docker cp metasim/cfg/robots/franka_cfg.py metasim_2:/home/lukesong_google_com/RoboVerse/metasim/cfg/robots/ && \
clear
