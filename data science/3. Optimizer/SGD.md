> **Algorithm 8.1** Stochastic gradient descent (SGD) update
> **Require**: Learning rate schedule $\epsilon_1, \epsilon_2, \ldots$ 
> **Require**: Initial parameter $\boldsymbol{\theta}$
> 	$k \leftarrow 1$
> 	**while** stopping criterion not met **do**
> 		- ==important==: Sample a minibatch of $m$ examples from the training set ${\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}}$ with corresponding targets $\boldsymbol{y}^{(i)}$. 
> 		- Compute gradient estimate: $\hat{\boldsymbol{g}} \leftarrow \frac{1}{m}\nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
> 		- Apply update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \epsilon_k \hat{\boldsymbol{g}}$
> 		- $k \leftarrow k + 1$
> 	**end while**