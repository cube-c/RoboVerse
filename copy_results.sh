#!/bin/bash
# copy_results.sh - fetch results from container

OBJ=${1:-"BBQ sauce"}   # default is "BBQ sauce", or pass another name like ./copy_results.sh "Ketchup"

# docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/LiberoPickChocolatePudding_${OBJ}_isaaclab.mp4" . && \
# docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/LiberoPickChocolatePudding_mujoco.mp4" . && \
# docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/img_with_point_LiberoPickChocolatePudding_${OBJ}.png" . && \
# docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/qwen2.5vl_top_one_LiberoPickChocolatePudding_${OBJ}_gsnet_visualization.png" .
docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/gsnet_top_one_LiberoPickChocolatePudding_gsnet_visualization.png" .
docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/world.ply" .
docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/LiberoPickChocolatePudding_isaaclab.mp4" .
docker cp metasim_2:"/home/lukesong_google_com/RoboVerse/vlm_manipulation/output/LiberoPickChocolatePudding_mujoco.mp4" .
clear
