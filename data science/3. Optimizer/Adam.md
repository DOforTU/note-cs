> **Algorithm 1**: Adam, our proposed algorithm for stochastic optimization
> **Require**: $\alpha$: Stepsize 
> **Require**: $\beta_1, \beta_2 \in [0, 1)$: Exponential decay rates for the moment estimates 
> **Require**: $f(\theta)$: Stochastic objective function with parameters $\theta$ 
> **Require**: $\theta_0$: Initial parameter vector
> 	- ==important==: $m_0 \leftarrow 0$ (Initialize 1st moment vector) 
> 	- ==important==: $v_0 \leftarrow 0$ (Initialize 2nd moment vector) 
> 	- $t \leftarrow 0$ (Initialize timestep)
> **while** $\theta_t$ not converged **do** 
> 	- $\quad t \leftarrow t + 1$ 
> 	- $\quad g_t \leftarrow \nabla_\theta f_t(\theta_{t-1})$ (Get gradients w.r.t. stochastic objective at timestep ) 
> 	- ==important==: $\quad m_t \leftarrow \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$ (Update biased first moment estimate) 
> 	- ==important==: $\quad v_t \leftarrow \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$ (Update biased second raw moment estimate) 
> 	- ==important==: $\quad \hat{m}_t \leftarrow m_t/(1 - \beta_1^t)$ (Compute bias-corrected first moment estimate) 
> 	- ==important==: $\quad \hat{v}_t \leftarrow v_t/(1 - \beta_2^t)$ (Compute bias-corrected second raw moment estimate) 
> 	- $\quad \theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$ (Update parameters) 
> **end while**
> **return** $\theta_t$ (Resulting parameters)

알고리즘 다른 표기법:

> **Algorithm 8.7** The Adam algorithm
> **Require**: Step size $\epsilon$ (Suggested default: 0.001) 
> **Require**: Exponential decay rates for moment estimates, $\rho_1$ and $\rho_2$ in $[0, 1)$. (Suggested defaults: 0.9 and 0.999 respectively) 
> **Require**: Small constant $\delta$ used for numerical stabilization. (Suggested default: $10^{-8}$) 
> **Require**: Initial parameters $\boldsymbol{\theta}$
> 	==important==: Initialize 1st and 2nd moment variables $\boldsymbol{s} = 0, \boldsymbol{r} = 0$ Initialize time step $t = 0$
> 	**while** stopping criterion not met **do**
> 		- Sample a minibatch of $m$ examples from the training set ${\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}}$ with corresponding targets $\boldsymbol{y}^{(i)}$.
> 		- Compute gradient: $\boldsymbol{g} \leftarrow \frac{1}{m}\nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
> 		- $t \leftarrow t + 1$
> 		- ==important==: Update biased first moment estimate: $\boldsymbol{s} \leftarrow \rho_1 \boldsymbol{s} + (1 - \rho_1)\boldsymbol{g}$
> 		- ==important==: Update biased second moment estimate: $\boldsymbol{r} \leftarrow \rho_2 \boldsymbol{r} + (1 - \rho_2)\boldsymbol{g} \odot \boldsymbol{g}$
> 		- ==important==: Correct bias in first moment: $\hat{\boldsymbol{s}} \leftarrow \frac{\boldsymbol{s}}{1 - \rho_1^t}$
> 		- ==important==: Correct bias in second moment: $\hat{\boldsymbol{r}} \leftarrow \frac{\boldsymbol{r}}{1 - \rho_2^t}$
> 		- Compute update: $\Delta\boldsymbol{\theta} = -\epsilon \frac{\hat{\boldsymbol{s}}}{\sqrt{\hat{\boldsymbol{r}} + \delta}}$ (operations applied element-wise)
> 		- Apply update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \Delta\boldsymbol{\theta}$
> 	**end while**