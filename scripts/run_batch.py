# scripts/run_batch.py
"""
Backward-compatible wrapper for the batch runner.

Prefer:
  tetrakis-batch --help
once installed via pip.
"""

from tetrakis_sim.batch import main

if __name__ == "__main__":
    main()
