# Dry Dock Detector

# Goal
Develop a sidescan sonar based detector for the Beckholmen dry dock environment

# Limitations
- Currently, SAM is limited to very basic automatic operations or teleoperation at the surface.
- Limited ground truth is available 

# Assumptions
- Operations will be at the surface, so the altitude will be constant.

# Approaches

## Approach A: ML
Experiment with deep learning methods to classify the sss returns.
I'm currently unsure about the details, but th will be filled in here as they become available.

### Open Questions
- Should I use the raw returns on maybe the down range gradients that were used for the buoy and rope detectors
- Process channels together or separately
