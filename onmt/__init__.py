""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.utils
import sys
import onmt.utils.optimizers
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.utils]

__version__ = "1.1.1"
