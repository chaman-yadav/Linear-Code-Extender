# Linear Extension & Generator Matrix Tool

A Streamlit web application that computes the minimal linear extension (span) of a set of binary vectors over GF(2) and derives the corresponding generator matrix for the linear code.

## Deployment on Streamlit Cloud

This application is deployed on Streamlit Cloud and publicly available at:  
🔗 [https://linear-code-extender.streamlit.app/](https://linear-code-extender.streamlit.app/)

## Features

- Computes all linear combinations (span) of input binary vectors using XOR operations
- Performs Gaussian elimination over GF(2) to find a basis
- Generates the canonical generator matrix from the basis vectors
- Verifies the generator matrix by showing all possible codewords
- Interactive web interface with input validation

## Usage

1. Enter the number of vectors and their size
2. Input your binary vectors (one per line, space-separated elements)
3. Click "Compute" to:
   - See all codewords in the span (minimal linear extension)
   - View the generator matrix (basis vectors as rows)
   - See all codewords generated from the basis

## Requirements

- Python 3.x
- Streamlit
- NumPy

## Installation

```bash
pip install streamlit numpy

streamlit run Minimal_Extension.py
