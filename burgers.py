# Burgers equation.
# 09 / 14 / 2021
# Edgar A. M. O.

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, type_of_loss, mu, epochs):
        # x & t from boundary conditions:
        self.x_u = torch.tensor(
            X_u[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True
        )
        self.t_u = torch.tensor(
            X_u[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True
        )

        # x & t from collocation points:
        self.x_f = torch.tensor(
            X_f[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True
        )
        self.t_f = torch.tensor(
            X_f[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True
        )

        # boundary solution:
        self.u = torch.tensor(u, dtype=torch.float32)

        # null vector to test against f:
        self.null = torch.zeros((self.x_f.shape[0], 1))
        self.type_of_loss = type_of_loss
        self.mu = mu
        assert type_of_loss in ["Custom Loss Function", "LASSO Regularization", "Ridge Regularization"]

        # initialize net:
        self.create_net()
        # self.net.apply(self.init_weights)
        self.epochs = epochs
        # this optimizer updates the weights and biases of the net:
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=self.epochs,
            max_eval=self.epochs,
            history_size=50,
            tolerance_grad=1e-05,
            tolerance_change=0.5 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        # typical MSE loss (this is a function):
        self.loss = nn.MSELoss()

        # loss :
        self.ls = 0

        # iteration number:
        self.iter = 0

    def create_net(self):
        """net takes a batch of two inputs: (n, 2) --> (n, 1)"""
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
            m.bias.data.fill_(0.001)

    def net_u(self, x, t):
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        f = u_t + (u * u_x) - (nu * u_xx)

        return f

    def plot(self, time):
        """plot the solution on new data"""

        x = torch.linspace(-1, 1, 200)
        # Plot the predicted solution at each time
        # Create an array of (x,t) pairs
        xcol = x.reshape(-1, 1)
        tcol = torch.full_like(xcol, time)

        # Predict the solution u at the (x,t) pairs
        usol = self.net_u(xcol, tcol)

        # Plot the predicted solution
        fig = plt.figure(figsize=(7 ,4))
        ax = fig.add_subplot(111)
        window_size = int(time*10)+1
        weights = np.ones(window_size) / window_size
        smooth_y = np.convolve(u, weights, mode='same')
        ax.plot(x_,smooth_y,color='blue',label='True solution',linewidth=2)
        ax.plot(x.numpy(), usol.detach().numpy(), label='PINN solution',linestyle='--',color='red',linewidth=2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u$')
        ax.legend()
        plt.show()
        fig.savefig("Plots/plot.png")
        x = torch.linspace(-1, 1, 200)
        t = torch.linspace( 0, 1, 100)

        # x & t grids:
        X, T = torch.meshgrid(x, t)

        # x & t columns:
        xcol = X.reshape(-1, 1)
        tcol = T.reshape(-1, 1)

        # one large column:
        usol = self.net_u(xcol, tcol)

        # reshape solution:
        U = usol.reshape(x.numel(), t.numel())

        # transform to numpy:
        xnp = x.numpy()
        tnp = t.numpy()
        Unp = U.detach().numpy()

        # plot:
        fig = plt.figure(figsize=(9, 4.5))
        ax = fig.add_subplot(111)

        h = ax.imshow(Unp,
                      interpolation='nearest',
                      cmap='rainbow', 
                      extent=[tnp.min(), tnp.max(), xnp.min(), xnp.max()], 
                      origin='lower', aspect='auto', label='Heatmap of Predicted solution')
        ax.set_xlabel("x (in m)")
        ax.set_ylabel("t (in s)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        fig.savefig("Plots/heatmap.png")
        plt.show()

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u & f predictions:
        u_prediction = self.net_u(self.x_u, self.t_u)
        f_prediction = self.net_f(self.x_f, self.t_f)

        # losses:
        u_loss = self.loss(u_prediction, self.u)
        f_loss = self.loss(f_prediction, self.null)
        self.ls = (
            u_loss
            + f_loss
            + self.mu
            * sum(p.pow(2).sum() for p in self.net.parameters())
            * (self.type_of_loss == "Ridge Regularization")
            + self.mu
            * sum(p.abs().sum() for p in self.net.parameters())
            * (self.type_of_loss == "LASSO Regularization")
        )
        # derivative with respect to net's weights:
        self.ls.backward()

        my_bar.progress(int((self.iter)/self.epochs*100), text="Training")
        # increase iteration count:
        self.iter += 1
        # print report:
        if not self.iter % 100:
            print("Epoch: {0:}, Loss: {1:6.3f}".format(self.iter, self.ls)) 

        return self.ls

    def train(self):
        """training loop"""
        self.net.train()
        self.optimizer.step(self.closure)


if __name__ == "__main__":
    st.title('PINN Demo')
    # define custom CSS
    custom_css = """
        <style>
            .small-title {
                font-size: 30px;
            }
        </style>
    """

    # add custom CSS to the page
    st.markdown(custom_css, unsafe_allow_html=True)

    # your app code here
    st.write(r'''
    Hello! Welcome to PINN demo application. Here we will use a Physics Informed Neural Network to visualize solutions to the following viscous Burgers Equation:

    $u_t+uu_x-\frac{0.01}{\pi}u_{xx}=0$

    The initial condition and the boundary conditions are as follows:

    $u(1,t)=u(-1,t)=0; $
    $u(x,0)=-sin(\pi{x})$

    Please select the parameters of your choice from below.

    Click on "Start Training" button once you are done choosing your desired parameters.
    ''')
    nx = 101
    nt = 100000
    dx = 2.0 / (nx - 1)
    nu = .01 / np.pi

    x_ = np.linspace(-1, 1, nx)
    u = -np.sin(np.pi * x_)
    
    def burgers(u, t, nu):
        dx = 2.0 / (nx - 1)
        dudt = np.zeros(nx)
        dudt[1:-1] = -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx) + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return dudt
    
    # Create an input field for entering the value of time
    
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
    # Display the selected values
    # Plot the PINN solution for the selected time value
    # (replace this with your own code for plotting the PINN solution)
    nu = 0.01 / np.pi  # constant in the diff. equation
    N_u = 100  # number of data points in the boundaries
    N_f = 10000

    data = loadmat("burgers_shock.mat")
    # Extract x and t values from the data
    x = data["x"].flatten()[:, None]
    t = data["t"].flatten()[:, None]
    # Create a grid of points (x,t)
    X, T = np.meshgrid(x, t)
    # Flatten the grid and concatenate x and t values to create X_u_train
    X_u_train = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # Extract u values from the data and reshape them to create u_train
    u_train = np.real(data["usol"]).T.flatten()[:, None]
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])
    # Create collocation points X_f_train
    N_f = 10000
    X_f_train = lb + (ub - lb) * np.random.rand(N_f, 2)
    # Initialize the PINN with the training data
    training=False
    col1, col2, col3 = st.columns(3)
    # Create a dropdown menu for selecting the type of regularization and loss function
    reg_type = col1.selectbox('Select regularization type:', ['Custom Loss Function','LASSO Regularization','Ridge Regularization'])
    time_value = col2.number_input('Enter time value (in s):', value=0.2,min_value=0.0, max_value=0.5)
    num_epochs=col3.number_input("Number of Epochs: ",value=500)
    # Create a slider for adjusting the value of the regularization constant
    reg_value=0
    if reg_type!='Custom Loss Function':
        reg_value = st.slider('Select regularization constant:', 0.0, 1.0, 0.1)
    
    if st.button('Start Training'):
        training = not training
    dt = time_value / nt
    for n in range(nt):
        u = cash_karp(u,n*dt ,dt ,nu)
    pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train, reg_type, reg_value, num_epochs)
    
    if training:
        st.write(f'Regularization type: {reg_type}')
        if reg_type!='Custom Loss Function':
            st.write(f'Regularization value: {reg_value}')
        st.write(f'Time value: {time_value}')
        my_bar = st.progress(0, text="Training")
        pinn.train()
        pinn.plot(time_value)
        st.markdown('<h2 class="small-title">PINN plots</h2>', unsafe_allow_html=True)

        # Display the saved PNG image
        st.image("Plots/heatmap.png")
        st.image('Plots/plot.png')
        training=False