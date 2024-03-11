import shutil
import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

cp_tableau = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
has_latex = shutil.which('latexmk') is not None

def enable_matplotlib_latex():
    if has_latex:
        plt.rcParams['text.usetex'] = True #default=False (mathtext mode)
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'


def hfbox_pos(key, h, w, POS):
    xc,yc = POS[key]
    h = h/2
    w = w/2
    ret = np.array([(xc-w,yc-h),(xc+w,yc-h),(xc+w,yc+h),(xc-w,yc+h),(xc-w,yc-h)]).T
    return ret


def hfline_pos(k0, k1, x0, x1, POS, shift=(0,0)):
    pos0 = POS[k0]
    pos1 = POS[k1]
    ret0 = tuple([(y0*(1-x0)+y1*x0) for y0,y1 in zip(pos0,pos1)])
    ret1 = tuple([(y0*(1-x1)+y1*x1) for y0,y1 in zip(pos0,pos1)])
    if len(shift)==2: #(x,y):
        ret0 = ret0[0]+shift[0], ret0[1]+shift[0]
        ret1 = ret1[0]+shift[1], ret1[1]+shift[1]
    else: #TODO bug
        assert len(shift)==4 #(x0,y0,x1,y1)
        ret0 = ret0[0]+shift[0],ret0[1]+shift[1]
        ret1 = ret1[0]+shift[2],ret1[1]+shift[3]
    return ret0,ret1


def hf3line_pos(k0, k1, k2, pc, frac, POS):
    p0,p1,p2 = [POS[x] for x in [k0,k1,k2]]
    # frac = (0.3, 0.35, 0.82)
    line = np.array([
        (p0[0]*(1-frac[0]) + pc[0]*frac[0], p0[1]*(1-frac[0]) + pc[1]*frac[0]),
        pc,
        (np.nan, np.nan),
        (p1[0]*(1-frac[1]) + pc[0]*frac[1], p1[1]*(1-frac[1]) + pc[1]*frac[1]),
        pc,
        (pc[0]*(1-frac[2]/2)+p2[0]*frac[2]/2, pc[1]*(1-frac[2]/2)+p2[1]*frac[2]/2),
    ]).T
    arrow = pc, (pc[0]*(1-frac[2]) + p2[0]*frac[2], pc[1]*(1-frac[2]) + p2[1]*frac[2])
    return line, arrow


def plot_qobject_trivialization_map(use_latex=True):
    if use_latex:
        enable_matplotlib_latex()
    text_kw = dict(verticalalignment='center', horizontalalignment='center', fontsize=16)
    colorq = cp_tableau[1]
    fig,ax = plt.subplots(figsize=(8,5))
    # https://stackoverflow.com/a/44528296/7290857 fancy arrow

    hf_circle_pos = lambda n: (6*np.cos(np.pi/2+2*np.pi*n/9), 5*np.sin(np.pi/2+2*np.pi*n/9))
    POS = {
        'Rn': (-1.2,0),
        'channel': hf_circle_pos(0),
        'stiefel': hf_circle_pos(1),
        'SU': hf_circle_pos(2),
        'su': hf_circle_pos(3),
        'herm': hf_circle_pos(4),
        'herm+': hf_circle_pos(5),
        'R+': (hf_circle_pos(5)[0], 0),
        'Delta': (2*hf_circle_pos(5)[0], 0),
        'L+': (hf_circle_pos(5)[0], 0.5*hf_circle_pos(5)[1]),
        'rho': hf_circle_pos(6),
        'psi': hf_circle_pos(7),
        'Sn': hf_circle_pos(8),
    }
    hfline = functools.partial(hfline_pos, POS=POS)
    hfbox = functools.partial(hfbox_pos, POS=POS)
    hf3line = functools.partial(hf3line_pos, POS=POS)

    ax.text(*POS['Rn'], 'Euclidean\nSpace', **text_kw)
    ax.add_patch(matplotlib.patches.Ellipse(POS['Rn'], width=2.7, height=2, facecolor="none", edgecolor="k"))
    ax.text(*POS['channel'], r'$\mathcal{E}_{d\to\hat{d},r}$', **text_kw, color=colorq)
    ax.text(*POS['stiefel'], r'$\mathrm{St}(n,r)$', **text_kw)
    ax.text(*POS['SU'], r'$\mathrm{SU}(n)$', **text_kw, color=colorq)
    ax.text(*POS['su'], r'$\mathfrak{su}(n)$', **text_kw)
    ax.text(*POS['herm'], r'$\mathrm{Herm}^{(n)}$', **text_kw, color=colorq)
    ax.text(*POS['herm+'], r'$\mathrm{Herm}_+^{(n,r)}$', **text_kw)
    ax.text(*POS['R+'], r'$\mathbb{R}_+$', **text_kw)
    ax.text(*POS['Delta'], r'$\Delta_+^n$', **text_kw)
    ax.text(*POS['L+'], r'$L_+^{(n,r)}$', **text_kw)
    ax.text(*POS['rho'], r'$\rho$', **text_kw, color=colorq)
    ax.text(*POS['psi'], r'$|\psi\rangle$', **text_kw, color=colorq)
    ax.text(*POS['Sn'], r'$S^n$', **text_kw)
    ax.plot(*hfbox('psi', 0.8, 0.8), color=colorq, linewidth=1)
    ax.plot(*hfbox('rho', 0.7, 0.7), color=colorq, linewidth=1)
    ax.plot(*hfbox('herm', 1.0, 1.7), color=colorq, linewidth=1)
    ax.plot(*hfbox('SU', 0.95, 1.35), color=colorq, linewidth=1)
    ax.plot(*hfbox('channel', 1.0, 1.4), color=colorq, linewidth=1)

    text_kw['fontsize']=11
    text_kw['color'] = cp_tableau[0]
    arrow_kw = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color=text_kw['color'])

    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'Sn', 0.21, 0.93), **arrow_kw))
    ax.text(1.5, 2.4, r'$(\sin,\cos)$', **text_kw, rotation=25)
    ax.text(1.7, 1.7, r'$\frac{x}{\| x \|_2}$', **text_kw, rotation=25)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'R+', 0.45, 0.92), **arrow_kw))
    ax.text(0.9, 0.3, r'SoftPlus', **text_kw)
    ax.text(0.9, -0.3, r'exp', **text_kw)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('R+', 'Delta', 0.13, 0.85), **arrow_kw))
    ax.text(3, 0.3, r'SoftMax', **text_kw)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('R+', 'L+', 0.13, 0.85), **arrow_kw))
    ax.text(2.7, -1.2, 'fill lower\npart', **text_kw)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Sn', 'psi', 0.08, 0.85), **arrow_kw))
    ax.text(5.0, 2.5, r'$x_1+ix_2$', **text_kw, rotation=-45)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('L+', 'herm+', 0.12, 0.87), **arrow_kw))
    ax.text(2.7, -3.2, r'Cholesky', **text_kw)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('herm+', 'rho', 0.2, 0.87), **arrow_kw))
    ax.text(4.0, -3.8, r'$\frac{x}{\mathrm{Tr}[x]}$', **text_kw, rotation=30)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('herm', 'herm+', 0.24, 0.75), **arrow_kw))
    ax.text(0, -4.45, r'exp, $n=r$', **text_kw)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('herm', 'su', 0.25, 0.85, shift=(0.05,0.05)), **arrow_kw))
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('su', 'herm', 0.14, 0.75, shift=(-0.05,-0.05)), **arrow_kw))
    ax.text(-3.5, -3.3, r'$i(x-\mathrm{Tr}[x]I_n/n)$', **text_kw, rotation=-25)
    ax.text(-4.15, -3.7, r'$ix+\theta I_n$', **text_kw, rotation=-25)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'herm', 0.25, 0.9), **arrow_kw))
    tmp0 = dict(**text_kw)
    tmp0['fontsize'] -= 1
    ax.text(-1.85, -2.5, r'$x+x^\dagger + i(x-x^\dagger)$', **tmp0, rotation=78)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'su', 0.28, 0.87), **arrow_kw))
    ax.text(-3.7, -1, 'Gell-Mann\nbasis', **text_kw, rotation=25)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('su', 'SU', 0.1, 0.85), **arrow_kw))
    ax.text(-5.3, -1, 'exp', **text_kw, rotation=-70)
    ax.text(-5.8, -1, 'Cayley', **text_kw, rotation=-70)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('SU', 'stiefel', 0.2, 0.85), **arrow_kw))
    ax.text(-5, 2.5, 'column', **text_kw, rotation=45)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'stiefel', 0.25, 0.87), **arrow_kw))
    ax.text(-2.6, 2.4, 'polar', **text_kw, rotation=-45)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('stiefel', 'channel', 0.2, 0.8), **arrow_kw))
    ax.text(-2.1, 4.65, 'transpose', **text_kw, rotation=12)

    pc = POS['rho'][0], POS['Delta'][1]-0.5
    tmp0, tmp1 = hf3line('Delta', 'psi', 'rho', pc, (0.3,0.35,0.82))
    ax.plot(*tmp0, color=text_kw['color'])
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*tmp1, **arrow_kw))
    tmp0 = r'$\sum_{i}p_i|\psi_i\rangle\langle\psi_i|$'
    ax.text(6.0, -1, tmp0, **text_kw)

    ax.set_xlim(-6.8, 6.9)
    ax.set_ylim(-5.4, 5.7)
    ax.axis('off')
    fig.tight_layout()
    return fig,ax


# numqi.entangle.AutodiffCHAREE
def plot_cha_trivialization_map(use_latex=True):
    if use_latex:
        enable_matplotlib_latex()
    POS = {
        'Rn': (-2,0),
        'Sna': (1,1.5),
        'Snb': (1,0),
        'psia': (3.5,1.5),
        'psib': (3.5,0),
        'Delta': (2, -1.5),
        'cha': (7,0),
    }
    hfline = functools.partial(hfline_pos, POS=POS)
    hf3line = functools.partial(hf3line_pos, POS=POS)
    text_kw = dict(verticalalignment='center', horizontalalignment='center', fontsize=16)
    text_kw_arrow = dict(verticalalignment='center', horizontalalignment='center', fontsize=13, color=cp_tableau[0])
    arrow_kw = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color=text_kw_arrow['color'])

    fig,ax = plt.subplots(figsize=(8,2.8)) #(6,0.7)
    ax.text(*POS['Rn'], r'$\mathbb{R}^{(2d_A+2d_B+1)N}$', **text_kw)
    ax.add_patch(matplotlib.patches.Ellipse(POS['Rn'], width=2.2, height=1.0, facecolor="none", edgecolor="k"))
    ax.text(*POS['Sna'], r'$S^{(2d_A-1)\times N}$', **text_kw)
    ax.text(*POS['Snb'], r'$S^{(2d_B-1)\times N}$', **text_kw)
    ax.text(*POS['psia'], r'$\left\{ |\psi_A^{(i)}\rangle \right\}$', **text_kw)
    ax.text(*POS['psib'], r'$\left\{ |\psi_B^{(i)}\rangle \right\}$', **text_kw)
    ax.text(*POS['Delta'], r'$p\in\Delta_+^{N-1}$', **text_kw)
    ax.text(*POS['cha'], 'CHA', **text_kw)

    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'Sna', 0.24, 0.84, shift=(0.3,-0.1,-0.3,0.25)), **arrow_kw))
    ax.text(-0.6, 1.0, r'$\frac{x}{\| x \|_2}$', **text_kw_arrow, rotation=50)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'Snb', 0.38, 0.75), **arrow_kw))
    ax.text(-0.4, 0.2, r'$\frac{x}{\| x \|_2}$', **text_kw_arrow)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Rn', 'Delta', 0.24, 0.80, shift=(0.1,0,0.1,-0.3)), **arrow_kw))
    ax.text(-0.05, -1.1, 'SoftMax', **text_kw_arrow, rotation=-32)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Sna', 'psia', 0.28, 0.75), **arrow_kw))
    ax.text(2.3, 1.7, r'$x_1+ix_2$', **text_kw_arrow)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*hfline('Snb', 'psib', 0.28, 0.75), **arrow_kw))
    ax.text(2.3, 0.2, r'$x_1+ix_2$', **text_kw_arrow)

    pc = POS['psib'][0]+0.4*(POS['cha'][0]-POS['psib'][0]), POS['psib'][1]
    tmp0, tmp1 = hf3line('psia', 'psib', 'cha', pc, (0.3,0.45,0.82))
    tmp0[:,0] += np.array([0.2,0.3])
    tmp2, _ = hf3line('psib', 'Delta', 'cha', pc, (0.45,0.2,0.82))
    tmp2[:,3] += np.array([0.2,-0.3])
    ax.plot(*tmp0, color=text_kw_arrow['color'])
    ax.plot(*tmp2, color=text_kw_arrow['color'])
    ax.add_patch(matplotlib.patches.FancyArrowPatch(*tmp1, **arrow_kw))
    tmp0 = r'$\sum_i p_i|\psi_A^{(i)}\rangle\langle \psi_A^{(i)}| \otimes|\psi_B^{(i)}\rangle\langle\psi_B^{(i)}| $'
    ax.text(5.5, -0.9, tmp0, **text_kw_arrow)

    ax.set_xlim(-3.3, 7.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig,ax
