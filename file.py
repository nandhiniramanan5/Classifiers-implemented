from __future__ import division  # floating point division
import math
import numpy as np

n= math.exp(-(math.pow(1e-3,2)/(2*math.pow(1e-4,2))))
print (1 / (math.sqrt(2*math.pi) * (1e-3)) * n)