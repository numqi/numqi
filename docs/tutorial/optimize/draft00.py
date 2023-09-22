import numpy as np
import matplotlib.pyplot as plt
import torch

import numqi

np_rng = np.random.default_rng()
hf_uniform_para = lambda *x: torch.nn.Parameter(torch.tensor(np_rng.uniform(-1, 1, size=x), dtype=torch.float64))

def hf_demo(model, num_repeat=3, xlim=None, ylim=None, x_optim=None, tag_logscale=False):
    theta_optim,info = numqi.optimize.minimize(model, tol=1e-10, num_repeat=num_repeat, tag_record_path=True)
    path = np.stack(info['path'])
    print(f'optimal theta: {theta_optim.x}')
    print(f'optimal loss: {theta_optim.fun}')

    hf_model = numqi.optimize.hf_model_wrapper(model)
    xdata = np.linspace(*xlim, 100)
    ydata = np.linspace(*ylim, 100)
    if len(x_optim)==2:
        zdata = np.array([[hf_model(np.array([x, y]), tag_grad=False) for x in xdata] for y in ydata])
    else:
        zdata = np.array([[hf_model(np.concatenate([np.array([x, y]),x_optim[2:]]), tag_grad=False) for x in xdata] for y in ydata])
    if tag_logscale:
        zdata = np.log(np.maximum(1e-4,zdata))
    fig,ax = plt.subplots()
    ax.contourf(xdata, ydata, zdata, levels=15, cmap='winter')
    ax.plot([x_optim[0]], [x_optim[1]], 'rx', label='optimal')
    ax.plot(path[:,0], path[:,1], '-.', label='path')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    plt.close(fig)


class Rastrigin(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.x = hf_uniform_para(n)
        self.A = 10
        # solution [0,0,...,0], 0

    def forward(self):
        x = self.x
        loss = self.A*x.shape[0] + (x*x - self.A*torch.cos(2*np.pi*x)).sum()
        return loss

class Ackley(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = hf_uniform_para(2)
        # solution [0,0] 0

    def forward(self):
        x,y = self.theta
        tmp0 = -20*torch.exp(-0.2*torch.sqrt(0.5*(x*x + y*y)))
        tmp1 = -torch.exp(0.5*(torch.cos(2*np.pi*x)+torch.cos(2*np.pi*y))) + np.e + 20
        ret = tmp0 + tmp1
        return ret


class Rosenbrock(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.theta = hf_uniform_para(n)
        # solution [1,1,...,1] 0

    def forward(self):
        tmp0 = self.theta[1:] - self.theta[:-1]**2
        tmp1 = 1-self.theta[:-1]
        ret = 100*torch.dot(tmp0, tmp0) + torch.dot(tmp1,tmp1)
        return ret


class Beale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [3,0.5] 0

    def forward(self):
        x,y = self.theta
        ret = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return ret

class GoldsteinPrice(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [0,-1] 0

    def forward(self):
        x,y = self.theta
        tmp0 = 1 + (x+y+1)**2 * (19-14*x+3*x*x-14*y+6*x*y+3*y*y)
        tmp1 = 30 + (2*x-3*y)**2 * (18-32*x+12*x*x+48*y-36*x*y+27*y*y)
        ret = tmp0*tmp1 - 3 #shift 3 for good plotting
        return ret


class Booth(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [1,3] 0

    def forward(self):
        x,y = self.theta
        tmp0 = x+2*y-7
        tmp1 = 2*x+y-5
        ret = tmp0*tmp0 + tmp1*tmp1
        return ret


class BukinFunctionN6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [-10,1] 0

    def forward(self):
        x,y = self.theta
        # possible numerical unstable here
        ret = 100*torch.sqrt(torch.abs(y-0.01*x*x)) + 0.01*torch.abs(x+10)
        return ret


class Matyas(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [-0,0] 0

    def forward(self):
        x,y = self.theta
        ret = 0.26*(x*x + y*y) - 0.48*x*y
        return ret


class LeviFunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [0,0] 0

    def forward(self):
        x,y = self.theta
        ret = torch.sin(3*np.pi*x)**2 + (x-1)**2*(1+torch.sin(3*np.pi*y)**2) + (y-1)**2*(1+torch.sin(2*np.pi*y)**2)
        return ret


class Himmelblau(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [3,2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126] 0

    def forward(self):
        x,y = self.theta
        tmp0 = x*x + y - 11
        tmp1 = x + y*y - 7
        ret = tmp0*tmp0 + tmp1*tmp1
        return ret


class ThreeHumpCamel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [0,0] 0

    def forward(self):
        x,y = self.theta
        ret = 2*x*x - 1.05*x*x*x*x + x*x*x*x*x*x/6 + x*y + y*y
        return ret

class Easom(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [pi,pi] 0

    def forward(self):
        x,y = self.theta
        ret = -torch.cos(x)*torch.cos(y)*torch.exp(-(x-np.pi)**2-(y-np.pi)**2)
        return ret

class CrossInTray(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [1.34941,-1.34941],[1.34941,1.34941],[-1.34941,1.34941],[-1.34941,-1.34941] -2.06261

    def forward(self):
        x,y = self.theta
        tmp0 = torch.abs(100 - torch.sqrt(x*x + y*y)/np.pi)
        tmp1 = torch.abs(torch.sin(x)*torch.sin(y)*torch.exp(tmp0))
        ret = -0.0001*tmp1**0.1
        return ret

class Eggholder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [512,404.2319] -959.6407

    def forward(self):
        x,y = self.theta
        tmp0 = -(y+47)*torch.sin(torch.sqrt(torch.abs(x/2 + (y+47))))
        tmp1 = -x*torch.sin(torch.sqrt(torch.abs(x-(y+47))))
        ret = tmp0 + tmp1
        return ret

class HolderTable(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [8.05502,9.66459] -19.2085

    def forward(self):
        x,y = self.theta
        tmp0 = torch.sin(x)*torch.cos(y)
        tmp1 = torch.exp(torch.abs(1 - torch.sqrt(x*x + y*y)/np.pi))
        ret = -torch.abs(tmp0*tmp1)
        return ret

class McCormick(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [-0.54719,-1.54719] -1.9133

    def forward(self):
        x,y = self.theta
        tmp0 = torch.sin(x+y)
        tmp1 = (x-y)**2
        tmp2 = -1.5*x + 2.5*y + 1
        ret = tmp0 + tmp1 + tmp2
        return ret

class SchafferFunctionN2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [0,0] 0

    def forward(self):
        x,y = self.theta
        tmp0 = torch.sin(x*x-y*y)
        tmp1 = 1 + 0.001*(x*x + y*y)
        ret = 0.5 + (tmp0*tmp0 - 0.5)/(tmp1*tmp1)
        return ret

class SchafferFunctionN4(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = hf_uniform_para(2)
        # [0,1.25313] 0.292579

    def forward(self):
        x,y = self.theta
        tmp0 = torch.cos(torch.sin(torch.abs(x*x-y*y)))
        tmp1 = 1 + 0.001*(x*x + y*y)
        ret = 0.5 + (tmp0*tmp0 - 0.5)/(tmp1*tmp1)
        return ret

class StyblinskiTangFunction(torch.nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.theta = hf_uniform_para(n)
        # [-2.903534]*n -39.16599*n

    def forward(self):
        x = self.theta
        ret = 0.5*(x*x*x*x - 16*x*x + 5*x).sum()
        return ret

n = 2
model = Rastrigin(n)
x_optim = np.zeros(n)
hf_demo(model, num_repeat=10, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12), x_optim=x_optim)
# when n is large, it's almost impossible to find the global minimum

model = Ackley()
x_optim = np.zeros(2)
hf_demo(model, num_repeat=10, xlim=(-5, 5), ylim=(-5, 5), x_optim=x_optim)

n = 2
model = Rosenbrock(n)
x_optim = np.ones(n)
hf_demo(model, num_repeat=10, xlim=(-2, 2), ylim=(-1, 3), x_optim=x_optim, tag_logscale=True)

model = Beale()
x_optim = np.array([3, 0.5])
hf_demo(model, num_repeat=10, xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), x_optim=x_optim, tag_logscale=True)

model = GoldsteinPrice()
x_optim = np.array([0, -1])
hf_demo(model, num_repeat=10, xlim=(-2, 2), ylim=(-3, 1), x_optim=x_optim, tag_logscale=True)

model = Booth()
x_optim = np.array([1, 3])
hf_demo(model, num_repeat=10, xlim=(-10, 10), ylim=(-10, 10), x_optim=x_optim, tag_logscale=True)

model = BukinFunctionN6()
x_optim = np.array([-10, 1])
# even bad for adam
hf_demo(model, num_repeat=10, xlim=(-15, -5), ylim=(-4, 6), x_optim=x_optim, tag_logscale=True)

model = Matyas()
x_optim = np.array([0, 0])
hf_demo(model, num_repeat=10, xlim=(-10, 10), ylim=(-10, 10), x_optim=x_optim, tag_logscale=True)

model = LeviFunction()
x_optim = np.array([0, 0])
hf_demo(model, num_repeat=10, xlim=(-10, 10), ylim=(-10, 10), x_optim=x_optim)

model = Himmelblau()
x_optim = np.array([[3, 2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]])
hf_demo(model, num_repeat=10, xlim=(-5, 5), ylim=(-5, 5), x_optim=x_optim[0], tag_logscale=True)

model = ThreeHumpCamel()
x_optim = np.array([0, 0])
hf_demo(model, num_repeat=10, xlim=(-5, 5), ylim=(-5, 5), x_optim=x_optim, tag_logscale=True)

model = Easom()
x_optim = np.array([np.pi, np.pi])
hf_demo(model, num_repeat=10, xlim=(-100, 100), ylim=(-100, 100), x_optim=x_optim)

model = CrossInTray()
x_optim = np.array([[1.34941,-1.34941],[1.34941,1.34941],[-1.34941,1.34941],[-1.34941,-1.34941]])
hf_demo(model, num_repeat=10, xlim=(-10, 10), ylim=(-10, 10), x_optim=x_optim[0])

model = Eggholder()
x_optim = np.array([512, 404.2319])
hf_demo(model, num_repeat=10, xlim=(-1000, 1000), ylim=(-1000, 1000), x_optim=x_optim)

model = HolderTable()
x_optim = np.array([[8.05502, 9.66459], [-8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, -9.66459]])
hf_demo(model, num_repeat=10, xlim=(-10, 10), ylim=(-10, 10), x_optim=x_optim[1])

# we initial at uniform (-1,1) which is not good for some functions

# out of domain
model = McCormick()
x_optim = np.array([-0.54719,-1.54719])
hf_demo(model, num_repeat=10, xlim=(-2.5, 4), ylim=(-3, 4), x_optim=x_optim)

model = SchafferFunctionN2()
x_optim = np.array([0, 0])
hf_demo(model, num_repeat=10, xlim=(-50, 50), ylim=(-50, 50), x_optim=x_optim)

model = SchafferFunctionN4()
x_optim = np.array([[0, 1.25313], [0, -1.25313], [1.25313,0], [-1.25313,0]])
hf_demo(model, num_repeat=10, xlim=(-50, 50), ylim=(-50, 50), x_optim=x_optim[0])

n = 2
model = StyblinskiTangFunction(n)
x_optim = np.array([-2.903534]*n)
hf_demo(model, num_repeat=10, xlim=(-5, 5), ylim=(-5, 5), x_optim=x_optim)
# TODO
