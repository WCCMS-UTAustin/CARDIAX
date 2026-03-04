
# NeoHookean (WORK IN PROGRESS)

## NeoHookean Equation

After looking at a simple theoretically nicer problem, [Poisson](poisson.md), we move towards the base of cardiac mechanics, hyperelasticity. This is where the power of the GPU based computations really shine because of the many iterations needed to solve these nonlinear problems. The starting point is the conservation of linear momentum

$$
\rho_0 \ddot{\varphi} = \nabla \cdot \mathbf{P} + \rho_0 \mathbf{b}
$$

To unpack the notation, $\varphi$ represents the motion of the body. Dynamics will be gone over in the [dynamics tutorial](../Advanced/dynamic_problems.md), so we will assume quasi-static. Thus, $\varphi$ is independent of time and for further simplification, assume no body force $\mathbf{b} = 0$ to get

$$
\nabla \cdot \mathbf{P} = 0
$$

We can obtain a displacement field $\mathbf{u} = \varphi(\mathbf{X}) - \mathbf{X}$ with $\mathbf{X}$ as the reference coordinate, where $\mathbf{u}$ will be the variable of interest. Then we have the deformation gradient $\mathbf{F}(\mathbf{X}) = \nabla \mathbf{\varphi} = \nabla \mathbf{u} + \mathbf{I}$. To obtain the First Piola Kirchhoff stress tensor, we need to define the strain energy which is where this becomes hyperelasticity. We define the strain energy as a Neo-Hookean

$$
\Psi(\mathbf{F}) = c_1 (\mathbf{F}^T \mathbf{F} - 3 - 2\ln(J)) + c_2 (J - 1)^2
$$

where $J = \det(\mathbf{F})$. Now differentiating with respect to $\mathbf{F}$ will give us

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
$$

The strong form is now written as 

$$
\nabla \cdot \frac{\partial \Psi}{\partial \mathbf{F}} = 0
$$

however, we'd prefer to solve the weak form as a boundary value problem, also referred to as virtual work

$$
\int_{\Omega_0} \mathbf{P} : \nabla \mathbf{v} dV = \int_{\Gamma} \mathbf{t} \cdot \mathbf{v} dS
$$

Now we have the contraction of two tensors that can occur on a per element level and a boundary condition that can be applied to the desired face. Now let's look into how we solve these problems.

Since the discretization steps are beyond these tutorials, we've only done them for the most basic example and will focus on the theoretics and code.