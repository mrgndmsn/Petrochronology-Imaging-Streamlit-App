from copy import deepcopy
import numpy as np
import pandas as pd
from .models import MapLayer
from .provenance import compatible_layers
from .analysis import evaluate_equation

"""Explicit dataset operations shared by the UI and calculation tests."""

