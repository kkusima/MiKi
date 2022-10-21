import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from jax import jacfwd
from autograd import jacobian, hessian

import jax
import jax.numpy as jnp


print('\n ONE EXAMPLE')
def f(x):
  return jnp.asarray(
    [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])

print("-"*50)
print('USING JAX')
print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))

print('\n TRYING OUT DIFFERENT AUTODIFF solvers \n')

def fun(x, a):
    return (x[0] - 1) **2 + (x[1] - a) **2

print("-"*50)
print('\n USING JAX')
def fun_der(x, a):
    return jacfwd(lambda x: fun(x, a))(x).ravel()

def fun_hess(x, a):
    return hessian(lambda x: fun(x, a))(x)

x0 = np.array([2, 0]) # initial guess
a = 2.5

res = minimize(fun, x0, args=(a,), method='dogleg', jac=fun_der, hess=fun_hess)
print((res))

print('\n')
print("-"*50)
print('USING AUTOGRAD')
import numpy as np
from scipy.optimize import minimize
from autograd import jacobian, hessian

def fun_der(x, a):
    return jacobian(lambda x: fun(x, a))(x).ravel()

def fun_hess(x, a):
    return hessian(lambda x: fun(x, a))(x)

x0 = np.array([2, 0]) # initial guess
a = 2.5

res = minimize(fun, x0, args=(a,), method='dogleg', jac=fun_der, hess=fun_hess)
print((res))

print('\n')
print("-"*50)
print('USING NUMDIFFTOOLS')
import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian

def fun_der(x, a):
    return Jacobian(lambda x: fun(x, a))(x).ravel()

def fun_hess(x, a):
    return Hessian(lambda x: fun(x, a))(x)

x0 = np.array([2, 0]) # initial guess
a = 2.5

res = minimize(fun, x0, args=(a,), method='dogleg', jac=fun_der, hess=fun_hess)
print((res))
