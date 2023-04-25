import numpy as np
import matplotlib.pyplot as plt

def burgers(u, t, nu):
    dx = 2.0 / (nx - 1)
    dudt = np.zeros(nx)
    dudt[1:-1] = -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx) + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return dudt

def cash_karp(u, t, dt, nu):
    c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])
    a = np.array([[0, 0, 0, 0, 0],
                  [1/5, 0, 0, 0, 0],
                  [3/40, 9/40, 0, 0, 0],
                  [3/10, -9/10, 6/5, 0, 0],
                  [-11/54, 5/2, -70/27, 35/27, 0],
                  [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])
    b = np.array([37/378, 0, 250/621, 125/594, 0 ,512/1771])
    bs = np.array([2825/27648 ,0 ,18575/48384 ,13525/55296 ,277/14336 ,1/4])
    
    k = np.zeros((6,nx))
    for i in range(6):
        k[i] = burgers(u + dt * np.dot(a[i,:i], k[:i]), t + c[i] * dt , nu)
    unew = u + dt * np.dot(b,k)
    return unew

nx = 101
nt = 100000
dx = 2.0 / (nx - 1)
nu = .01 / np.pi

x = np.linspace(-1, 1, nx)
u = -np.sin(np.pi * x)

t_plot = float(input('Enter time to plot: '))
dt = t_plot / nt

for n in range(nt):
    u = cash_karp(u,n*dt ,dt ,nu)

fig = plt.figure(figsize=(7 ,4))
ax = fig.add_subplot(111)
window_size = int(t_plot*10)+1
weights = np.ones(window_size) / window_size
smooth_y = np.convolve(u, weights, mode='same')
ax.plot(x,smooth_y,color='blue')
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
plt.show()
