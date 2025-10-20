> **Algorithm 8.2** Stochastic gradient descent (SGD) with momentum
> **Require**: Learning rate $\epsilon$, momentum parameter $\alpha$. **Require**: Initial parameter $\boldsymbol{\theta}$, initial velocity $\boldsymbol{v}$.
> **while** stopping criterion not met **do**
> 	- Sample a minibatch of $m$ examples from the training set ${\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}}$ with corresponding targets $\boldsymbol{y}^{(i)}$.
> 	- Compute gradient estimate: $\boldsymbol{g} \leftarrow \frac{1}{m}\nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
> 	- ==important==: Compute velocity update: $\boldsymbol{v} \leftarrow \alpha \boldsymbol{v} - \epsilon \boldsymbol{g}$ 
> 	- Apply update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{v}$
> **end while**