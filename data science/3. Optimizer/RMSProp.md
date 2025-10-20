> **Algorithm 8.5** The RMSProp algorithm
> **Require**: Global learning rate $\epsilon$, decay rate $\rho$. **Require**: Initial parameter $\boldsymbol{\theta}$ 
> **Require**: Small constant $\delta$, usually $10^{-6}$, used to stabilize division by small numbers.
> 	Initialize accumulation variables $\boldsymbol{r} = 0$
> 	**while** stopping criterion not met **do**
> 		- Sample a minibatch of $m$ examples from the training set ${\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}}$ with corresponding targets $\boldsymbol{y}^{(i)}$.
> 		- Compute gradient: $\boldsymbol{g} \leftarrow \frac{1}{m}\nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
> 		- ==important==: Accumulate squared gradient: $\boldsymbol{r} \leftarrow \rho \boldsymbol{r} + (1 - \rho)\boldsymbol{g} \odot \boldsymbol{g}$
> 		- Compute parameter update: $\Delta\boldsymbol{\theta} = -\frac{\epsilon}{\sqrt{\delta + \boldsymbol{r}}} \odot \boldsymbol{g}$. ($\frac{1}{\sqrt{\delta + \boldsymbol{r}}}$ applied element-wise)
> 		- Apply update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \Delta\boldsymbol{\theta}$
> 	**end while**