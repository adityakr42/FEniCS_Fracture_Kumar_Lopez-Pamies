# FEniCS_Fracture_Kumar_Lopez-Pamies
! This repository contains an example code in FEniCS to solve a boundary value problem using the new fracture phase-field model introduced in [1]. The example problem studied is the Single edge notch tension test in alumina. This phase-field model that can describe fracture nucleation in all settings, be it from large pre-existing cracks, small pre-existing cracks,
smooth and non-smooth boundary points, or within the bulk of structures subjected to arbitrary quasistatic loadings, while keeping undisturbed the ability of the standard phase-field formulation to model crack propagation. The model involves the addition of an external configurational force $ce$ to the evolution equation for phase-field variable $z$. This external force depends on the strength surface of the material. In this work, we assume strength surface as the Drucker-Prager surface defined by the tensile and compressive strength. Additionally, the force involves a parameter $delta$ that needs to be calibrated to describe fracture nucleation (propagation) from large pre-existing cracks. See Section 4.3.2 in [1] for more details. We have provided an additional code in this repository 'Calibration_of_ce.py' that automatically calibrates the value of the parameter delta. See 'Readme_Calibration.txt' for more details on the calibration.

!**********************************************************************
! Usage:
!
! This code takes as an input of 5 material properties listed below:
! 1. E = Young's modulus
! 2. nu = Poisson's ratio
! 3. Gc = Critical energy release rate
! 4. sts = Tensile strength
! 5. scs = Compressive strength

! Additionally it needs as an input the value of regularization length $eps$ that the user wants to adopt in this boundary value problem. Typically, $eps$ should be chosen so that it much smaller than the average size of the structure. Also, $eps$ should not be chosen to have a larger value than the Irwin's characteristic length (E*Gc/(sts^2)). Otherwise the nucleation (or propagation) from medium-length pre-existing cracks can't be described properly. However, $eps$ can be chosen to be much smaller than the Irwin's characteristic length. Also note, the effective regularization length is even smaller than $eps$ for positive values of $delta$ as defined in equation (21) in [1]. For this chosen value of $eps$, one can determine the value of parameter $delta$ using the code 'Calibration_of_ce.py' in this repository. The output of that code will also tell the user what mesh size they should use in the regions the fracture will happen.

! For this boundary value problem of SENT test, the code will give as output the critical value of stress at which the pre-existing crack propagates (or nucleates or initiates).

!**********************************************************************
! References:

! [1] Kumar, A., Bourdin, B., Francfort, G.A. and Lopez-Pamies, O., 2020. 
      Revisiting nucleation in the phase-field approach to brittle fracture. 
      J. Mech. Phys. Solids, 104027. 
