---
title: Home
layout: home
---

# CS 6476 Group 60 Project Update

## Introduction
We are investigating the problem of how of accurately ball actions from a video feed of a soccer game using state-of-the-art computer vision methods. Examples of ball actions include passes, drives, crosses and many others.

Being able to accurately identify such ball actions can be extremely useful. For example, if one does not have access to a video stream of the game due to slow internet access, they can still follow an automated live commentary feed to follow the important plays in a game. In addition, such information can augment a live broadcast of a soccer game by automating the process of keeping a camera's focus on not only where the ball is, but the entire region of interest based on the ball action currently being performed.

Such a method can also be extrapolated beyond the arena of sports to efficiently find instances of any action in video.

## Related Works

There has been many approaches used to perform action spotting and other related tasks. For example, when optimizing for speed and efficient processing, linear support vector machines have been used. SVMs are comparatively easy to implement on simple hardware, useful in robotics applications where compute is limited. This method also introduces a novel way of encoding temporal information in a compact way called HMHH. HMHH improves on MHI (motion history image) which sums together the current frame along with a few past frames with decaying weights. Unfortunately, this means that when a pixel is important in both an old and the current frame, the information from the old frame is mostly clobbered. HMHH allows us to preserve more of this information in a compact way.

Another approach is through reinforcement learning.
## Methods/Approach



## Experiemnts/Results

## What's Next?

## Contributions
