import numpy as np
import matplotlib.pyplot as plt
from util import load_image
import diff_ops
import util

def total_variation_denoising(image, lambda_param=0.05, tau_param=0.125, max_iterations=150, rof_epsilon=1e-8):
    # input image is color png depicting grayscale, use first plane from here
    image = image[:, :, 1].astype(np.float64)

    u = image
    convergence_epsilon = 1e-4

    for iteration in range(max_iterations):

        # store old u for convergence check
        u_old = u

        # compute the gradients, magnitude of gradients and normalized gradients of u
        # take into account the rof_epsilon when normalizing
        grad_u_x = diff_ops.dx_forward(u)
        grad_u_y = diff_ops.dy_forward(u)
        mag_grad_u = np.sqrt(np.multiply(grad_u_x, grad_u_x) + np.multiply(grad_u_y, grad_u_y))
        grad_u_x = grad_u_x / (mag_grad_u + rof_epsilon)
        grad_u_y = grad_u_y / (mag_grad_u + rof_epsilon)

        # compute the laplace of u
        modified_laplace_u = (diff_ops.dx_backward(grad_u_x) + diff_ops.dy_backward(grad_u_y))

        # Gradient Decent Optimization iteration of u
        u = u - tau_param * (-modified_laplace_u + lambda_param * (u - image))

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # print("Tikhonov loss: {}".format(re))

        # compare the change of u with the stopping criteria
        if re < convergence_epsilon:
            break

    return u


def tikhnov_denoising(image, lambda_param=0.05, tau_param=0.125, max_iterations=50, convergence_epsilon=1e-8):

    # input image is color png depicting grayscale, use first plane from here
    image = image[:, :, 1].astype(np.float64)

    u = image

    # gradient optimization loop
    for iteration in range(max_iterations):

        # store old u for convergence check
        u_old = u

        # compute the gradient of u
        grad_u_x = diff_ops.dx_forward(u)
        grad_u_y = diff_ops.dy_forward(u)

        # compute the laplace of u
        laplace_u = diff_ops.dx_backward(grad_u_x) + diff_ops.dy_backward(grad_u_y)

        # Gradient Descent Optimization
        u = u - tau_param * (-laplace_u + lambda_param * (u - image))

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # compare the change of u with the stopping criteria
        if re < convergence_epsilon:
            break
    return u


