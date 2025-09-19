# MATLAB Scripts (Optional)

⚠️ **These MATLAB scripts are OPTIONAL** - The repository includes a complete Python implementation.

## Quick Summary

The `matlab/` directory contains legacy MATLAB scripts that were used to develop the original RLMPC implementation. These are preserved for reference only.

**You do NOT need MATLAB to use this repository.** The Python implementation in `src/` provides all functionality with better performance and no licensing requirements.

## Contents

- `mpc_setup/` - Original MPC setup and system parameters
- `smpc_training/` - Legacy data generation (replaced by Python)
- `analysis/` - Analysis and plotting tools

## Recommendation

Use the Python pipeline documented in the main [README.md](../README.md). The MATLAB scripts are only useful for:
- Comparing implementations
- Legacy workflow migration
- Research reference

**For new projects, use Python exclusively.**