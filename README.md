# FEniCS_Fracture_Kumar_Lopez-Pamies
! This FEniCS code finds the paramter $delta$ in the configurational external force $c_e$ introduced in [1]. The configurational force is part of a newly 
proposed phase-field model that can describe fracture nucleation in all settings, be it from large pre-existing cracks, small pre-existing cracks,
smooth and non-smooth boundary points, or within the bulk of structures subjected to arbitrary quasistatic loadings, while keeping undisturbed the ability of the standard phase-field formulation to model crack propagation.
The parameter $delta$ needs to be calibrated to describe fracture nucleation (propagation) from large pre-existing cracks. See Section 4.3.2 in [1] for more details. In short, this code solves the classical problem of center cracked test specimen in LEFM and the phase-field solution is compared with the analytical solution.

!**********************************************************************
! Usage:
!
! This code requires an input of 5 material properties listed below:
! 1. E = Young's modulus
! 2. nu = Poisson's ratio
! 3. Gc = Critical energy release rate
! 4. sts = Tensile strength
! 5. scs = Compressive strength

! Also needed as an input is the value of regularization length $eps$ that the user wants to adopt in the boundary value problem that they wish to use. Typically, $eps$ should be chosen so that it much smaller than the average size of the structure. Also, $eps$ should not be chosen to have a larger value than the Irwin's characteristic length (E*Gc/(sts^2)). Otherwise the nucleation (or propagation) from medium-length pre-existing cracks can't be described properly. However, $eps$ can be chosen to be much smaller than the Irwin's characteristic length. Also note, the effective regularization length is even smaller than $eps$ for positive values of $delta$ as defined in equation (21) in [1].

! The code will give as output the calibrated value of $delta$ for the chosen value of epsilon. It also gives the value of the mesh size $h$ that one should adopt. Moreover, it provides the value of the effective regularization length $eps*$ and ratio of $eps*$ to mesh size $h$. It is left to user to check that the ratio of $eps*$ to $h$ is atleast greater than 3. Please choose a higher value of $eps$ if that's not the case.

!**********************************************************************
! References:

! [1] Kumar, A., Bourdin, B., Francfort, G.A. and Lopez-Pamies, O., 2020. 
      Revisiting nucleation in the phase-field approach to brittle fracture. 
      J. Mech. Phys. Solids, 104027. 
