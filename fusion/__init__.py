"""fusion -- Data Fusion Interview Challenge pipeline (see /CLAUDE.md)."""
import torch

# CLAUDE.md §2.4: "torch.set_default_dtype(torch.float64) (double, like C/R)" -- set once here
# so every fusion.* module and every test gets double precision on import, with no per-file repeats.
torch.set_default_dtype(torch.float64)

