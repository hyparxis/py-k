# py-k
These are some simple demos of my various experiments doing IK via constrained optimization. I noticed there aren't a lot of resources on IK for systems closed loops/holonomic constraints out there, so I hope these examples can serve as inspiration for others.

For ease of prototyping I'm using the excellent HIPS autograd library to automatically compute Jacobians, but note that for any real-world use of these algorithms the Jacobians can (and should) be computed analytically.
