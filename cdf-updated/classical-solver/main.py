import numpy as np 
from matplotlib import pyplot as plt 


# Grid points 

nx =  41
ny =  41

nt = 500 # time steps 
nit = 50 # pressure step 

# Grid spacing

dx = 2/(nx-1)
dy = 2/(ny-1)

rho =1 
nu = 0.1
dt = 0.001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))


def build_b(b, u, v, dx, dy, dt, rho):

    b[1:-1,1:-1] = rho * (1/(2*dt)) * ( (u[1:-1, 2:] - u[1:-1, 0:-2]) * dx + (v[2:, 1:-1] - v[0:-2, 1:-1]) * dy )
    return b 

def pressure_poisson(p, dx, dy, b):

    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1: -1, 1: -1] = (pn[:-2, 1:-1] + pn[2:, 1:-1]) * dy**2 + (pn[1:-1, :-2] + pn[1:-1, 2:]) * dx**2 - b[1:-1, 1:-1] * dx**2 * dy**2 / (2 * (dx**2 + dy**2))
        p[:, -1] = p[:, -2] # wall boundary condition
        p[0, :] = p[1, :] # wall boundary condition
        p[:, 0] = p[:, 1] # wall boundary condition
        p[-1, :] = 0 # wall boundary condition
    return p


def cavity_flow(u,v,p):

    un = np.empty_like(u)
    vn = np.empty_like(v)

    for n in range(nt):

        un = u.copy()
        vn = v.copy()

        b = build_b(b,u,v,dx,dy,dt,rho)
        p = pressure_poisson(p,dx,dy,b)

        u[1:-1,1:-1] = un[1:-1,1:-1] \
            - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2]) \
            - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1]) \
            - dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2]) \
            + nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]) +
                  dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))

        v[1:-1,1:-1] = vn[1:-1,1:-1] \
            - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2]) \
            - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1]) \
            - dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1]) \
            + nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2]) +
                  dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1]))

        u[0,:]=0
        u[:,0]=0
        u[:,-1]=0
        u[-1,:]=1

        v[0,:]=0
        v[-1,:]=0
        v[:,0]=0
        v[:,-1]=0

    return u,v,p


velocity = np.sqrt(u**2 + v**2)
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
X,Y = np.meshgrid(x,y)

velocity = np.sqrt(u**2 + v**2)

plt.figure(figsize=(10,6))

plt.contourf(X,Y,velocity, cmap="jet")
plt.colorbar(label="Velocity magnitude")

plt.streamplot(X,Y,u,v,color="black")

plt.title("Flow Field")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

plt.figure(figsize=(10,6))

# Pressure field
plt.contourf(p, alpha=0.6)
plt.colorbar(label="Pressure")

# Velocity vectors
plt.quiver(u, v)

plt.title("Velocity and Pressure Field")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()