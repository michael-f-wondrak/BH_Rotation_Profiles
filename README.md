# BH_Rotation_Profiles
Determining Profiles of Characteristic Angular Velocities from Radio-Image Sequences of Black Holes

Authors (alphabetically): Rohan G. Amanaganti, Bram van den Berg, and Michael F. Wondrak

Adapted and extended program based on
- Cylinder_Clean.py (N. Conroy, M. Baubock, C. Gammie, https://doi.org/10.5281/zenodo.13717347):
  "A cleaned version of the script used in 'Rotation in Event Horizon Telescope Movies', by Conroy et al. 2023"
- patternspeed.py (R. Dahale and EHT Collaboration, https://github.com/rohandahale/ehteval/commit/f01c5793b1963bfa478f4925528a8fd358d39d17)
  "Validation and evaluation of horizon-scale Sagittarius A* video reconstructions"

Determining patten speed profiles (characteristic angular velocities) from videos of black holes in general relativity and beyond.
Input is consecutive covariantly ray-traced snapshots from covariant magnetohydrodynamic simulations and the position and size of the bright ring including uncertainties.
Various ray-tracer file formats (as produced by BHOSS, ipole, RAPTOR) as well as Fits can be read in.
For information on usage of this script, call this script with the argument --help.
