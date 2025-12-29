# Scripts

This directory contains entry-point scripts and utilities for running the fraud detection pipeline.

- **`run_data_pipeline.py`**: Orchestrates the full end-to-end data processing pipeline by calling modules in `src/`.
- **`create_test_data.py`**: Generates synthetic test data for CI/CD and unit testing.

Core processing logic has been moved to the `src/` directory to follow industry standards for Python project structure.
