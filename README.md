# Synchronizer & Visualizer
This repo is intended to use python's multi-thread processing mechanism to do visualization and synchronization between a prophesee camera and an ORBBEC Femote Bolt RGBD camera. And this can be further used to create a reconstruction pipeline using an event and rgbd camera.

## Hardware setting
As prophesee EVK4 camera support hardware trigger, ORBBEC camera support this hardware trigger as well. Thus we use this software as a complement with the stm32f103 made hardware trigger. The trigger is working at 30fps

## What's more
I will keep refreshing and completing this repo to support more features. Any suggestions are kindly welcomed. And the synchronization logic and problem actually fit when you are synchronizing any Frame-Based camera with an Event-Based camera.
