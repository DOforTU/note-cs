> **Algorithm 8.4** The AdaGrad algorithm
> **Require**: Global learning rate $\epsilon$ **Require**: Initial parameter $\boldsymbol{\theta}$ 
> **Require**: Small constant $\delta$, perhaps $10^{-7}$, for numerical stability
> 	==important==: Initialize gradient accumulation variable $\boldsymbol{r} = \boldsymbol{0}$
> 	**while** stopping criterion not met **do**
> 		- Sample a minibatch of $m$ examples from the training set ${\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}}$ with corresponding targets $\boldsymbol{y}^{(i)}$.
> 		- Compute gradient: $\boldsymbol{g} \leftarrow \frac{1}{m}\nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
> 		- ==important==: Accumulate squared gradient: $\boldsymbol{r} \leftarrow \boldsymbol{r} + \boldsymbol{g} \odot \boldsymbol{g}$
> 		- ==important==: Compute update: $\Delta\boldsymbol{\theta} \leftarrow -\frac{\epsilon}{\delta + \sqrt{\boldsymbol{r}}} \odot \boldsymbol{g}$. (Division and square root applied element-wise)
> 		- Apply update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \Delta\boldsymbol{\theta}$
> 	**end while**