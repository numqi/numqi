# special state

## Werner state

1. reference
   * [wiki/werner-state](https://en.wikipedia.org/wiki/Werner_state)
   * [quantiki/werner-state](https://www.quantiki.org/wiki/werner-state)
   * [PRA88.032323](http://dx.doi.org/10.1103/PhysRevA.88.032323) Compatible quantum correlations: Extension problems for Werner and isotropic states

$$ \rho_{W,d}\left(a\right)=\frac{d-a}{d\left(d^{2}-1\right)}I+\frac{ad-1}{d\left(d^{2}-1\right)}\sum_{ij}\left|ij\right\rangle \left\langle ji\right|,\quad a\in\left[-1,1\right] $$

$$ \rho_{W,d}^{\prime}\left(\alpha\right)=\frac{1}{d^{2}-d\alpha}I-\frac{\alpha}{d^{2}-d\alpha}\sum_{ij}\left|ij\right\rangle \left\langle ji\right|,\quad\alpha\in\left[-1,1\right] $$

$$ \rho_{W,d}\left(a\right)=\rho_{W,d}^{\prime}\left(\frac{1-ad}{d-a}\right),\quad\rho_{W,d}\left(\frac{1-\alpha d}{d-\alpha}\right)=\rho_{W,d}^{\prime}\left(\alpha\right) $$

| $a$ | $\alpha$ | state |
| :-: | :-: | :-: |
| 1 | -1 | $xI+y\sum_{ij}\lvert ij\rangle\langle ji\rvert$ |
| $1/d$ | 0 | $I/d^2$ |
| 0 | $1/d$ | xx |
| -1 | 1 | xx |

SEP boundary

$$ a\in\left[0,1\right],\alpha\in\left[-1,\frac{1}{d}\right] $$

(1,k) extension boundary

$$ \left(1,k\right)\;\mathrm{ext}:\quad a\in\left[\frac{1-d}{k},1\right],\alpha\in\left[-1,\frac{k+d^{2}-d}{kd+d-1}\right] $$

## Isotropic state

1. reference
   * [quantiki/isotropic-state](https://www.quantiki.org/wiki/isotropic-state)
   * [PRA88.032323](http://dx.doi.org/10.1103/PhysRevA.88.032323) Compatible quantum correlations: Extension problems for Werner and isotropic states

$$ \rho_{I,d}\left(a\right)=\frac{d-a}{d\left(d^{2}-1\right)}I+\frac{ad-1}{d\left(d^{2}-1\right)}\sum_{i}\left|ii\right\rangle \left\langle ii\right|,\quad a\in\left[0,d\right] $$

$$ \rho_{I,d}^{\prime}\left(\alpha\right)=\frac{1-\alpha}{d^{2}}I+\frac{\alpha}{d}\sum_{i}\left|ii\right\rangle \left\langle ii\right|,\quad\alpha\in\left[-\frac{1}{d^{2}-1},1\right] $$

$$ \rho_{I,d}\left(\frac{1+\alpha d^{2}-\alpha}{d}\right)=\rho_{I,d}^{\prime}\left(\alpha\right),\quad\rho_{I,d}\left(a\right)=\rho_{I,d}^{\prime}\left(\frac{ad-1}{d^{2}-1}\right) $$

| $a$ | $\alpha$ | state |
| :-: | :-: | :-: |
| 0 | $-\frac{1}{d^2-1}$ | xx |
| $1/d$ | 0 | $I/d$ |
| $d$ | 1 | $\sum_{i}\lvert ii\rangle\langle ii\rvert$ |

SEP boundary

$$ a\in\left[0,1\right],\alpha\in\left[-\frac{1}{d^{2}-1},\frac{1}{d+1}\right] $$

(1,k) extension boundary

$$ a\in\left[0,1+\frac{d-1}{k}\right],\alpha\in\left[-\frac{1}{d^{2}-1},\frac{kd+d^{2}-d-k}{k\left(d^{2}-1\right)}\right] $$

## maximum entangled state

TODO
