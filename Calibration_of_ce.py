###########################################################################################################################################################################################################################
# This FEniCS code finds the configurational force $c_e$ introduced in  Kumar, A., Bourdin, B., Francfort, G.A. and Lopez-Pamies, O., 2020. "Revisiting nucleation in the phase-field approach to brittle fracture". Journal of the Mechanics and Physics of Solids, 104027.
# The configurational force represents the entire strength surface, parametrized here through the tensile strength and compressive strength.
# Specifically, this code calibrates the parameter $delta$ (a function of regularization length epsilon) in $c_e$ that controls the nucleation (or propagation) from large-preexisting cracks.
#
# The calibration of $delta$ is done through following the procedure outlined in Section 4.3.2 of the same paper.
# In short, the classical problem of center cracked test specimen in LEFM is solved a few times and phase-field solution is compared with the analytical solution.
#
# Input: Set the material properties in lines 18-20 and set the regularization length epsilon in line 24.
#
# Output: The fitted value of $delta$ along with the value of the effective regularization length $eps*$ will be printed in the file 'fitted_delta.txt'.
#
# Contact Aditya Kumar (akumar51@illinois.edu) for questions.
###########################################################################################################################################################################################################################

from dolfin import *
import numpy as np
import time

# Material properties
E, nu = 9800, 0.13	#Young's modulus and Poisson's ratio
Gc= 0.091125	#Critical energy release rate
sts, scs= 27, 77	#Tensile strength and compressive strength
#Irwin characteristic length
lch=3*Gc*E/8/(sts**2)
#The regularization length
eps=0.15   #epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work


#Don't need to change anything from here on.
########################################################

set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

#Geometry
ac=int(lch*40.0) #20.0
W, L = ac*5, ac*15

# Create mesh
if eps>lch:
	epsbyh=5
elif eps>lch/2.0 and eps<=lch:
	epsbyh=7.5
elif eps>lch/4.0 and eps<=lch/2.0:
	epsbyh=10.0
else:
	epsbyh=15.0

h=eps/epsbyh
mesh=RectangleMesh(comm, Point(0.0,0.0), Point(W,L), int(W/(32*h)), int(L/(32*h)))
domain1 = CompiledSubDomain("x[1]<lch*4", lch=lch)
ir=0
while ir<2:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain1.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

domain2 = CompiledSubDomain("x[1]<2.5*eps && x[0]<a+h*8*4 && x[0]>a-h*8*4", a=ac, eps=eps)
ir=0
while ir<3:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain2.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

# Choose phase-field model
phase_model=1;  #1 for AT1 model, 2 for AT2 model


# Define function space
V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-4)
front =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = W, tol=1e-4)
top =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = L, tol=1e-4)
bottom = CompiledSubDomain("x[1]<1e-4 && x[0]>a-1e-2", a=ac)
cracktip = CompiledSubDomain("x[1]<1e-4 && x[0]>a-h*8*2 && x[0]<a+1e-4 ", a=ac)
outer= CompiledSubDomain("x[1]>L/10", L=L)	
	
# Define Dirichlet boundary conditions
c=Expression("t*0.0",degree=1,t=0)
								
bcl = DirichletBC(V.sub(0), c, left )
bct = DirichletBC(V.sub(1), c, bottom)
bcs = [bcl, bct]

cz=Constant(1.0)
bct_z = DirichletBC(Y, cz, outer)
cz2=Constant(0.0)
bct_z2 = DirichletBC(Y, cz2, cracktip)
bcs_z=[bct_z, bct_z2]

# Define Neumann boundary conditions
sigma_external=1.25*sqrt(E*Gc/np.pi/ac)/((1-0.025*(ac/W)**2+0.06*(ac/W)**4)/sqrt(np.cos(np.pi*ac/2/W)))
Tf  = Expression(("t*0.0", "t*sigma"),degree=1,t=0, sigma=sigma_external)  # Traction force on the top boundary 
# marking boundary on which Neumann bc is applied
boundary_subdomains = MeshFunction("size_t", mesh, 1)
boundary_subdomains.set_all(0)
top.mark(boundary_subdomains,1)	
ds = ds(subdomain_data=boundary_subdomains) 

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
u_inc = Function(V)
dz = TrialFunction(Y)            # Incremental phase field
y  = TestFunction(Y)             # Test function
z  = Function(Y)                 # Phase field from previous iteration
z_inc = Function(Y)
d = u.geometric_dimension()


#Initialisation of displacement field,u and the phase field,z
u_init = Constant((0.0,  0.0))
u.interpolate(u_init)
for bc in bcs:
	bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
	bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))	
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))
	

u_prev = Function(V)
assign(u_prev,u)
z_prev = Function(Y)
assign(z_prev,z)
	
#Label the dofs on boundary
def extract_dofs_boundary(V, bsubd):	
	label = Function(V)
	label_bc_bsubd = DirichletBC(V, Constant((1,1)), bsubd)
	label_bc_bsubd.apply(label.vector())
	bsubd_dofs = np.where(label.vector()==1)[0]
	return bsubd_dofs

#Dofs on which reaction is calculated
top_dofs=extract_dofs_boundary(V,top)
y_dofs_top=top_dofs[1::d]


#Function to evaluate a field at a pint
def evaluate_function(u, x):
	comm = u.function_space().mesh().mpi_comm()
	if comm.size == 1:
		return u(*x)

	# Find whether the point lies on the partition of the mesh local
	# to this process, and evaulate u(x)
	cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
	u_eval = u(*x) if distance < DOLFIN_EPS else None

	# Gather the results on process 0
	comm = mesh.mpi_comm()
	computed_u = comm.gather(u_eval, root=0)

	# Verify the results on process 0 to ensure we see the same value
	# on a process boundary
	if comm.rank == 0:
		global_u_evals = np.array([y for y in computed_u if y is not None], dtype=np.double)
		assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

		computed_u = global_u_evals[0]
	else:
		computed_u = None

	# Broadcast the verified result to all processes
	computed_u = comm.bcast(computed_u, root=0)

	return computed_u



##Strain Energy, strain and stress functions in linear isotropic elasticity
 
mu, lmbda, kappa = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(E/(3*(1 - 2*nu)))

def energy(v):
	return mu*(inner(sym(grad(v)),sym(grad(v))) + ((nu/(1-nu))**2)*(tr(sym(grad(v))))**2 )+  0.5*(lmbda)*(tr(sym(grad(v)))*(1-2*nu)/(1-nu))**2 
	
def epsilon(v):
	return sym(grad(v))

def sigma(v):
	return 2.0*mu*sym(grad(v)) + (lmbda)*tr(sym(grad(v)))*(1-2*nu)/(1-nu)*Identity(len(v))

def sigmavm(sig,v):
	return sqrt(1/2*(inner(sig-1/3*tr(sig)*Identity(len(v)), sig-1/3*tr(sig)*Identity(len(v))) + (1/9)*tr(sig)**2 ))


eta=0.0
# Stored strain energy density (compressible L-P model)
psi1 =(z**2+eta)*(energy(u))	
psi11=energy(u)
# Total potential energy
Pi = psi1*dx - dot(Tf, u)*ds(1)
# Compute first variation of Pi (directional derivative about u in the direction of v)
R = derivative(Pi, u, v) 

# Compute Jacobian of R
Jac = derivative(R, u, du)

###First run of the code
if eps>lch:
	delta1=0.25
elif eps>lch/2.0 and eps<=lch:
	delta1=1.0
elif eps>lch/4.0 and eps<=lch/2.0:
	delta1=5.0
else:
	delta1=15.0

delta=delta1
beta0=-3*Gc/8/eps*delta	
beta3=0*(eps*sts)/(mu*kappa*Gc)
beta1= ((9*E*Gc)/2. - (9*E*Gc*sts)/(2.*scs) - (9*beta3*E*Gc*scs*sts)/2. + (9*beta3*E*Gc*sts**2)/2.)/(24.*E*eps*sts) +  (-12*beta0*E + (12*beta0*E*sts)/scs + 12*scs*sts + 12*beta3*scs**3*sts - 12*sts**2 - 12*beta3*sts**4)/(24.*E*sts)
beta2= (9*E*Gc*scs + 9*E*Gc*sts + 9*beta3*E*Gc*scs**2*sts + 9*beta3*E*Gc*scs*sts**2)/(16.*sqrt(3)*E*eps*scs*sts) +  (-24*beta0*E*scs - 24*beta0*E*sts - 24*scs**2*sts - 24*beta3*scs**4*sts - 24*scs*sts**2 - 24*beta3*scs*sts**4)/(16.*sqrt(3)*E*scs*sts)

pen=1000*conditional(lt(-beta0,Gc/eps),Gc/eps, -beta0)

ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0)/(1+beta3*(z**4)*(tr(sigma(u)))**2)

#To use later for memory allocation for these tensors
A=PETScMatrix()
b=PETScVector()

#Balance of configurational forces PDE
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=20*pen/2*( 1/4*( abs(z_prev+0.0005-z)-(z_prev+0.0005-z) )**2 )*dx
if phase_model==1:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) #+ y*(ce)*dx #linear model
else:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+ Gc*(y*(z-1)/eps + eps*inner(grad(z),grad(y)))*dx #+ derivative(Wv2,z,y)  #quadratic model
	
# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)

# Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "cg",   
                                          "preconditioner": "amg",						  
                                          "maximum_iterations": 10,
                                          "report": True,
                                          "error_on_nonconvergence": False}}			


#time-stepping parameters
T=1
Totalsteps=500
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-9

while t-stepsize < T:

	if comm_rank==0:
		print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
	
	c.t=t; Tf.t=t;
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<100 and rnorm_stag > 1e-8:
		start_time=time.time()
		##############################################################
		#First PDE
		##############################################################		
		Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
		solver_u  = NonlinearVariationalSolver(Problem_u)
		solver_u.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_u.solve()
			
		##############################################################
		#Second PDE
		##############################################################
		Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
		solver_z  = NonlinearVariationalSolver(Problem_z)
		solver_z.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_z.solve()
			
		min_z = z.vector().min();
		zmin = MPI.min(comm, min_z)
		if comm_rank==0:
			print(zmin)
		
		if comm_rank==0:
			print("--- %s seconds ---" % (time.time() - start_time))

	  
		###############################################################
		#Residual check for stag loop
		###############################################################
		b=assemble(-R, tensor=b)
		fint=b.copy() #assign(fint,b) 
		for bc in bcs:
			bc.apply(b)
		rnorm_stag=b.norm('l2')	
		stag_iter+=1  

		
	######################################################################
	#Post-Processing
	######################################################################
	assign(u_prev,u)
	assign(z_prev,z)
	
	####Calculate Reaction
	Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
	z_x = evaluate_function(z, (ac+eps,0.0))
	if comm_rank==0:
		print(Fx)
		print(z_x)
		with open('/fenics/fit2.txt', 'a') as rfile: 
			rfile.write("%s %s %s\n" % (str(t), str(zmin), str(z_x)))
	
	if z_x<0.1:
		t1=t
		break
	
	if t>0.997:
		t1=1
	
	#time stepping
	step+=1
	t+=stepsize

#########################################################################	
#########################################################################
# Second run
#########################################################################
#########################################################################
t=0
c.t=t; Tf.t=t;
##############################################################
#Initialisation of displacement field,u and the phase field,z
##############################################################
u_init = Constant((0.0,  0.0))
u.interpolate(u_init)
for bc in bcs:
	bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
	bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))	
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))
	

u_prev = Function(V)
assign(u_prev,u)
z_prev = Function(Y)
assign(z_prev,z)

#Parameters-second run
if t1>0.8 and t1<1:
	delta2=delta1/2
elif t1<0.8 and t1>0.6:
	delta2=delta1*2
elif t1<0.6 and t1>0.4:	
	delta2=delta1*8
elif t1<0.4:	
	delta2=delta1*20
elif t1==1:
	delta2=delta1/10

delta=delta2
beta0=-3*Gc/8/eps*delta	
beta1= ((9*E*Gc)/2. - (9*E*Gc*sts)/(2.*scs) - (9*beta3*E*Gc*scs*sts)/2. + (9*beta3*E*Gc*sts**2)/2.)/(24.*E*eps*sts) +  (-12*beta0*E + (12*beta0*E*sts)/scs + 12*scs*sts + 12*beta3*scs**3*sts - 12*sts**2 - 12*beta3*sts**4)/(24.*E*sts)
beta2= (9*E*Gc*scs + 9*E*Gc*sts + 9*beta3*E*Gc*scs**2*sts + 9*beta3*E*Gc*scs*sts**2)/(16.*sqrt(3)*E*eps*scs*sts) +  (-24*beta0*E*scs - 24*beta0*E*sts - 24*scs**2*sts - 24*beta3*scs**4*sts - 24*scs*sts**2 - 24*beta3*scs*sts**4)/(16.*sqrt(3)*E*scs*sts)


ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0)/(1+beta3*(z**4)*(tr(sigma(u)))**2)
#Balance of configurational forces PDE
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=20*pen/2*( 1/4*( abs(z_prev+0.0005-z)-(z_prev+0.0005-z) )**2 )*dx
if phase_model==1:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y)  #linear model
else:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+ Gc*(y*(z-1)/eps + eps*inner(grad(z),grad(y)))*dx  #quadratic model
	
# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)

#time-stepping parameters
T=1
Totalsteps=500
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-9

while t-stepsize < T:

	if comm_rank==0:
		print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
	
	c.t=t; Tf.t=t;
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<100 and rnorm_stag > 1e-8:
		start_time=time.time()
		##############################################################
		#First PDE
		##############################################################		
		Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
		solver_u  = NonlinearVariationalSolver(Problem_u)
		solver_u.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_u.solve()
			
		##############################################################
		#Second PDE
		##############################################################
		Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
		solver_z  = NonlinearVariationalSolver(Problem_z)
		solver_z.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_z.solve()
			
		min_z = z.vector().min();
		zmin = MPI.min(comm, min_z)
		if comm_rank==0:
			print(zmin)
		
		if comm_rank==0:
			print("--- %s seconds ---" % (time.time() - start_time))

	  
		###############################################################
		#Residual check for stag loop
		###############################################################
		b=assemble(-R, tensor=b)
		fint=b.copy() #assign(fint,b) 
		for bc in bcs:
			bc.apply(b)
		rnorm_stag=b.norm('l2')	
		stag_iter+=1  

		
	######################################################################
	#Post-Processing
	assign(u_prev,u)
	assign(z_prev,z)
	
	####Calculate Reaction
	Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
	z_x = evaluate_function(z, (ac+eps,0.0))
	if comm_rank==0:
		print(Fx)
		print(z_x)
		with open('/fenics/fit2.txt', 'a') as rfile: 
			rfile.write("%s %s %s\n" % (str(t), str(zmin), str(z_x)))
	
	if z_x<0.1:
		t2=t
		break
	
	#time stepping
	step+=1
	t+=stepsize


#########################################################################	
#########################################################################
# Third run
#########################################################################
#########################################################################
t=0
c.t=t; Tf.t=t;
##############################################################
#Initialisation of displacement field,u and the phase field,z
##############################################################
u_init = Constant((0.0,  0.0))
u.interpolate(u_init)
for bc in bcs:
	bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
	bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))	
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))
	

u_prev = Function(V)
assign(u_prev,u)
z_prev = Function(Y)
assign(z_prev,z)

#Parameters-Third run
if t1<1:
	delta3=delta1+(0.8-t1)/(t2-t1)*(delta2-delta1)
else:
	delta3=delta2*2

delta=delta3
beta0=-3*Gc/8/eps*delta	
beta1= ((9*E*Gc)/2. - (9*E*Gc*sts)/(2.*scs) - (9*beta3*E*Gc*scs*sts)/2. + (9*beta3*E*Gc*sts**2)/2.)/(24.*E*eps*sts) +  (-12*beta0*E + (12*beta0*E*sts)/scs + 12*scs*sts + 12*beta3*scs**3*sts - 12*sts**2 - 12*beta3*sts**4)/(24.*E*sts)
beta2= (9*E*Gc*scs + 9*E*Gc*sts + 9*beta3*E*Gc*scs**2*sts + 9*beta3*E*Gc*scs*sts**2)/(16.*sqrt(3)*E*eps*scs*sts) +  (-24*beta0*E*scs - 24*beta0*E*sts - 24*scs**2*sts - 24*beta3*scs**4*sts - 24*scs*sts**2 - 24*beta3*scs*sts**4)/(16.*sqrt(3)*E*scs*sts)


ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0)/(1+beta3*(z**4)*(tr(sigma(u)))**2)
#Balance of configurational forces PDE
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=20*pen/2*( 1/4*( abs(z_prev+0.0005-z)-(z_prev+0.0005-z) )**2 )*dx
if phase_model==1:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y)  #linear model
else:
	R_z = y*2*z*(psi11)*dx+ y*(ce)*dx+ Gc*(y*(z-1)/eps + eps*inner(grad(z),grad(y)))*dx  #quadratic model
	
# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)

#time-stepping parameters
T=1
Totalsteps=500
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-9

while t-stepsize < T:

	if comm_rank==0:
		print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
	
	c.t=t; Tf.t=t;
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<100 and rnorm_stag > 1e-8:
		start_time=time.time()
		##############################################################
		#First PDE
		##############################################################		
		Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
		solver_u  = NonlinearVariationalSolver(Problem_u)
		solver_u.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_u.solve()
			
		##############################################################
		#Second PDE
		##############################################################
		Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
		solver_z  = NonlinearVariationalSolver(Problem_z)
		solver_z.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_z.solve()
			
		min_z = z.vector().min();
		zmin = MPI.min(comm, min_z)
		if comm_rank==0:
			print(zmin)
		
		if comm_rank==0:
			print("--- %s seconds ---" % (time.time() - start_time))

	  
		###############################################################
		#Residual check for stag loop
		###############################################################
		b=assemble(-R, tensor=b)
		fint=b.copy() #assign(fint,b) 
		for bc in bcs:
			bc.apply(b)
		rnorm_stag=b.norm('l2')	
		stag_iter+=1  

		
	######################################################################
	#Post-Processing
	assign(u_prev,u)
	assign(z_prev,z)
	
	####Calculate Reaction
	Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
	z_x = evaluate_function(z, (ac+eps,0.0))
	if comm_rank==0:
		print(Fx)
		print(z_x)
		with open('/fenics/fit2.txt', 'a') as rfile: 
			rfile.write("%s %s %s\n" % (str(t), str(zmin), str(z_x)))
	
	if z_x<0.1:
		t3=t
		break
	
	#time stepping
	step+=1
	t+=stepsize


delta_fitted= delta3+(0.8-t3)/(t2-t3)*(delta2-delta3)
eps_star=eps/sqrt(1+delta_fitted)
epsstarbyh=eps_star/(eps/epsbyh)
if comm_rank==0:
	with open('/fenics/fitted_delta.txt', 'a') as rfile: 
		rfile.write("Calibrated value of delta= %s\n" % (str(delta_fitted)))
		rfile.write("Average value of mesh size= %s\n" % (str(eps/epsbyh)))
		rfile.write("epsilon_star= %s\n" % (str(eps_star)))
		rfile.write("Ratio of epsilon_star to h= %s\n" % (str(epsstarbyh)))
	print('Fitted value of delta= %f' %delta_fitted)


