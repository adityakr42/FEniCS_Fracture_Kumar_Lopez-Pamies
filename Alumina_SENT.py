###########################################################################################################################################################################################################################
# This FEniCS code implements the new phase-field model introduced in  Kumar, A., Bourdin, B., Francfort, G.A. and Lopez-Pamies, O., 2020. "Revisiting nucleation in the phase-field approach to brittle fracture". Journal of the Mechanics and Physics of Solids, 104027.
# This new phase-field model is used to solve the 'Single edge notch tension test' in Alumina.
# The key difference in this model compared to traditional phase-field models is the configurational external force $c_e$ that represents the strength surface of the material, parametrized here through the tensile strength and compressive strength.
#
#
# Input: Set the material properties in lines 19-21. Then set the regularization length epsilon in line 25. Also provide the value of delta and mesh size (in regions where fracture happens) calibrated using the code 'Calibration_of_ce.py' provided with this code.
#
# Output: The critical value of stress at which the pre-existing notch starts to propagate is printed in the file 'Critical_stress.txt'.
#
# Contact Aditya Kumar (akumar51@illinois.edu) for questions.
###########################################################################################################################################################################################################################

from dolfin import *
import numpy as np
import time

# Material properties
E, nu = 335000, 0.25	#Young's modulus and Poisson's ratio
Gc= 0.0268	#Critical energy release rate
sts, scs= 210, 2100	#Tensile strength and compressive strength
#Irwin characteristic length
lch=3*Gc*E/8/(sts**2)
#The regularization length
eps=0.04   #epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work
delta=6.73
h=0.004

# Problem description
comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

#Geometry of the single edge notch geometry
ac=0.25  #notch length
W, L = 6.0, 20.0  #making use of symmetry

# Create mesh
mesh=RectangleMesh(comm, Point(0.0,0.0), Point(W,L), int(W/(32*h)), int(L/(32*h)))
domain1 = CompiledSubDomain("x[1]<lch*4", lch=lch)
ir=0
while ir<2:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain1.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

domain2 = CompiledSubDomain("x[1]<2.5*eps && x[0]<a+h*8*4 && x[0]>a-h*8*4", a=ac, eps=eps, h=h)
ir=0
while ir<3:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain2.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-4)
front =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = W, tol=1e-4)
top =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = L, tol=1e-4)
bottom = CompiledSubDomain("x[1]<1e-4 && x[0]>a-1e-4", a=ac)
cracktip = CompiledSubDomain("x[1]<1e-4 && x[0]>a-h*8*2 && x[0]<a+1e-4 ", a=ac, h=h)
righttop = CompiledSubDomain("abs(x[1]-L)<1e-4 && abs(x[0]-W)<1e-4 ", L=L, W=W)
outer= CompiledSubDomain("x[1]>L/10", L=L)

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



# Choose phase-field model
phase_model=1;  #1 for AT1 model, 2 for AT2 model


# Define function space
V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z
	
# Define Dirichlet boundary conditions
c=Expression("t*0.0",degree=1,t=0)
								
bcl= DirichletBC(V.sub(0), Constant(0.0), righttop, method='pointwise'  )
bct = DirichletBC(V.sub(1), c, bottom)
bcs = [bcl, bct]

cz=Constant(1.0)
bct_z = DirichletBC(Y, cz, outer)
cz2=Constant(0.0)
bct_z2 = DirichletBC(Y, cz2, cracktip)
bcs_z=[bct_z, bct_z2]

# Define Neumann boundary conditions
sigma_external=1.05*sqrt(E*Gc/np.pi/ac)/((0.752+2.02*(ac/W)+0.37*(1-np.sin(np.pi*ac/2/W))**3)*(sqrt(2*W/np.pi/ac*np.tan(np.pi*ac/2/W)))/(np.cos(np.pi*ac/2/W)))
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
		with open('Alumina_SENT.txt', 'a') as rfile: 
			rfile.write("%s %s %s\n" % (str(t), str(zmin), str(z_x)))
	
	
	
	if z_x<0.1:
		t1=t
		break
	
	if t>0.997:
		t1=1
	
	#time stepping
	step+=1
	t+=stepsize


sigma_critical=t1*sigma_external
if comm_rank==0:
	with open('Critical_stress.txt', 'a') as rfile: 
		rfile.write("Critical stress= %s\n" % (str(sigma_critical)))
	print('Critical stress= %f' %sigma_critical)


