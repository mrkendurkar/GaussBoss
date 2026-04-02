# GaussBoss

      *       *
        *  *
         /\
        /  \   <- Gaussian “orbit curve”
   ∘   /    \
Asteroid ⊙

**Asteroid Detection and Orbit Determination Pipeline**

GaussBoss is a Python-based pipeline for identifying asteroids in astronomical images, tracking their motion, and generating MPC-formatted reports for orbit determination.

Designed for both amateur astronomers and professional observatories, it automates:

- Detection of moving objects across multi-epoch images
- Filtering false positives using velocity analysis
- MPC-compliant reporting with RA, Dec, and observation timestamps
- Support for customizable temporary designations and velocity annotations
## Example Output
T0001 C2026 04 01.81234 12 34 56.78 +12 34 56.7


## Installation

```bash
git clone https://github.com/<your-username>/GaussBoss.git
cd GaussBoss
pip install -r requirements.txt
