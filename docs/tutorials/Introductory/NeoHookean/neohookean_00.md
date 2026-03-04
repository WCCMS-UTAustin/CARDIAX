
# NeoHookean (WORK IN PROGRESS)

## NeoHookean Equation

After looking at a simple theoretically nicer problem, [Poisson](poisson.md), we move towards the base of cardiac mechanics, hyperelasticity. Due to the many iterations needed to solve the associated nonlinear problems, this is where the power of the GPU-based computations really shine. The starting point is the conservation of linear momentum,

$$
\rho_0 \ddot{\varphi} = \nabla \cdot \mathbf{P} + \rho_0 \mathbf{b}.
$$

To unpack the notation, $\varphi$ represents the motion of the body. Dynamics will be covered in [dynamics tutorial](../Advanced/dynamic_problems.md), while here we assume the body is quasi-static. By definition of quasi-static loading, $\varphi$ is independent of time, thus $\ddot{\varphi}=\mathbf{0}$. For further simplification, assume no body acceleration, $\mathbf{b} = 0$, to get

$$
\nabla \cdot \mathbf{P} = 0.
$$

We seek to obtain the displacement field $\mathbf{u} = \varphi(\mathbf{X}) - \mathbf{X}$, with $\mathbf{X}$ as the reference coordinate. The deformation gradient is given by $\mathbf{F}(\mathbf{X}) = \nabla \mathbf{\varphi} = \nabla \mathbf{u} + \mathbf{I}$. By definition of hyperelasticity, the first Piola-Kirchhoff stress tensor is entirely given by a derivative of the strain energy density $\Psi$. For a Neo-Hookean,

$$
\Psi(\mathbf{F}) = c_1 (\text{tr}({\mathbf{F}^T \mathbf{F}}) - 3 - 2\ln(J)) + c_2 (J - 1)^2,
$$

where $J = \det(\mathbf{F})$. Now differentiating with respect to $\mathbf{F}$ will give us

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}.
$$

The strong form is now written as 

$$
\nabla \cdot \frac{\partial \Psi}{\partial \mathbf{F}} = 0.
$$

For solving this differential equation by the finite element method, we'd prefer to solve the weak form, in this case referred to as virtual work,

$$
\int_{\Omega_0} \mathbf{P} : \nabla \mathbf{v} dV = \int_{\Gamma} \mathbf{t} \cdot \mathbf{v} dS,
$$

where \mathbf{t} is the traction vector on the boundary $\Gamma$ and $\mathbf{v}$ is a test function. Now we have the contraction of two tensors that can occur on a per-element level and a boundary condition that can be applied to the desired face. Now let's look into how we solve these problems.
