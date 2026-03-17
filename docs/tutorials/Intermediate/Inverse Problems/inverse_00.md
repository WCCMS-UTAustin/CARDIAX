# Inverse Problems: Theory

## PDE-Constrained Optimization

Let $\Omega \subset \mathbb{R}^d$ be a domain with boundary $\partial\Omega$. We consider a parameter-dependent PDE: given a parameter field $m \in \mathcal{P}$, find $u \in \mathcal{U}$ satisfying

$$
\mathcal{F}(u, m) = 0 \quad \text{in } \Omega, \qquad u = g \quad \text{on } \partial\Omega
$$

where $\mathcal{F}: \mathcal{U} \times \mathcal{P} \to \mathcal{V}^*$ is the residual operator mapping into the dual of the test space $\mathcal{V}$, and $g$ encodes the boundary data. The map $m \mapsto u(m)$ is defined implicitly by these equations. Given an objective functional $\mathcal{J}: \mathcal{U} \to \mathbb{R}$, the inverse problem is

$$
\min_{m \in \mathcal{P}} \; J(m) := \mathcal{J}(u(m))
$$

subject to $\mathcal{F}(u(m), m) = 0$. Gradient-based methods require $\mathrm{d}J/\mathrm{d}m$, which depends on the sensitivity of the state $u$ to the parameters.

## Continuous Adjoint Derivation

The Lagrangian is formed by augmenting the objective with the PDE constraint via a multiplier $\lambda \in \mathcal{V}$:

$$
\mathcal{L}(u, m, \lambda) = \mathcal{J}(u) + \langle \lambda, \mathcal{F}(u, m) \rangle
$$

where $\langle \cdot, \cdot \rangle$ denotes the duality pairing between $\mathcal{V}$ and $\mathcal{V}^*$. Since $\mathcal{F}(u(m), m) = 0$ when $u$ satisfies the PDE, $\mathcal{L} = J(m)$ for all $\lambda$. The total derivative with respect to $m$ is

$$
\frac{\mathrm{d}J}{\mathrm{d}m} = \frac{\partial \mathcal{J}}{\partial u} \frac{\mathrm{d}u}{\mathrm{d}m} + \left\langle \lambda, \frac{\partial \mathcal{F}}{\partial u} \frac{\mathrm{d}u}{\mathrm{d}m} + \frac{\partial \mathcal{F}}{\partial m} \right\rangle
$$

Collecting the terms involving $\mathrm{d}u/\mathrm{d}m$:

$$
\frac{\mathrm{d}J}{\mathrm{d}m} = \left(\frac{\partial \mathcal{J}}{\partial u} + \left\langle \lambda, \frac{\partial \mathcal{F}}{\partial u} \cdot \right\rangle \right) \frac{\mathrm{d}u}{\mathrm{d}m} + \left\langle \lambda, \frac{\partial \mathcal{F}}{\partial m} \right\rangle
$$

Choosing $\lambda$ to annihilate the first term — i.e., satisfying the **adjoint equation**

$$
\left(\frac{\partial \mathcal{F}}{\partial u}\right)^* \lambda = -\left(\frac{\partial \mathcal{J}}{\partial u}\right)^*
$$

where $(\cdot)^*$ denotes the adjoint operator — eliminates the $\mathrm{d}u/\mathrm{d}m$ term entirely. The reduced gradient is then

$$
\boxed{\frac{\mathrm{d}J}{\mathrm{d}m} = \left\langle \lambda, \frac{\partial \mathcal{F}}{\partial m} \right\rangle}
$$

The adjoint equation is a single linear PDE for $\lambda$, independent of the dimension of $\mathcal{P}$. This is what makes gradient-based PDE-constrained optimization tractable when $\dim \mathcal{P} \gg 1$.

## Discrete Setting

The state space $\mathcal{U}$ is replaced by a finite-dimensional subspace $\mathcal{U}^h \subset \mathcal{U}$ spanned by the FE basis, with $n_u$ DOFs. The parameter field is similarly discretized to $\mathbf{m} \in \mathbb{R}^{n_m}$. The PDE constraint becomes the assembled nonlinear algebraic system

$$
\mathbf{R}(\mathbf{u}, \mathbf{m}) = \mathbf{0}, \qquad \mathbf{R}: \mathbb{R}^{n_u} \times \mathbb{R}^{n_m} \to \mathbb{R}^{n_u}
$$

and the objective reduces to $J: \mathbb{R}^{n_u} \to \mathbb{R}$. The Lagrangian argument carries over directly, yielding the **discrete adjoint equation**

$$
\mathbf{K}^T \boldsymbol{\lambda} = -\left(\frac{\partial J}{\partial \mathbf{u}}\right)^T
$$

where $\mathbf{K} = \partial \mathbf{R}/\partial \mathbf{u} \in \mathbb{R}^{n_u \times n_u}$ is the tangent stiffness evaluated at the converged primal $\mathbf{u}^*$. The discrete reduced gradient is

$$
\frac{\mathrm{d}J}{\mathrm{d}\mathbf{m}} = \boldsymbol{\lambda}^T \frac{\partial \mathbf{R}}{\partial \mathbf{m}}
$$

The matrix $\partial \mathbf{R}/\partial \mathbf{m} \in \mathbb{R}^{n_u \times n_m}$ is never formed explicitly. It enters only through the vector-Jacobian product $\boldsymbol{\lambda}^T (\partial \mathbf{R}/\partial \mathbf{m})$, computed as a single reverse-mode AD pass over $\mathbf{m} \mapsto \mathbf{R}(\mathbf{u}^*, \mathbf{m})$ with cotangent $\boldsymbol{\lambda}$.

### Algorithm

Given a converged primal $\mathbf{u}^*$ and assembled $\mathbf{K}$:

1. Compute the adjoint RHS $-(\partial J / \partial \mathbf{u})^T$ analytically or via AD on $J$.
2. Solve $\mathbf{K}^T \boldsymbol{\lambda} = -(\partial J / \partial \mathbf{u})^T$.
3. Evaluate $\boldsymbol{\lambda}^T (\partial \mathbf{R}/\partial \mathbf{m})$ via a VJP of $\mathbf{m} \mapsto \mathbf{R}(\mathbf{u}^*, \mathbf{m})$.

Total cost: one additional linear solve of size $n_u$ and one VJP — both independent of $n_m$.

---

*Next: [CARDIAX Implementation](inverse_01.md)*
