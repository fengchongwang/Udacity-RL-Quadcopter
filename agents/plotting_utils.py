import numpy as np
import matplotlib.pyplot as plt
from agents.agent import OUNoise
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(task, ep_to_plot, pos_all, xmax=150, ymax=150, zmin=0, zmax=300, ax=None):
    """
    plot trajectories of given episodes
    
    ep_to_plot: list, episodes to plot
    pos_all: numpy array, shape (num_ep, max_t, 3), x, y, z positions
    xmax: x axis limits for plotting, [-xmax, xmax]
    ymax: y axis limits for plotting, [-ymax, ymax]
    zmin, zmax: z axis limits for plotting, [zmin, zmax]
    ax: plot on a specified ax (for subplot)
    """

    cmaps = ['Greens', 'Purples', 'Oranges', 'Blues', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
             'viridis', 'plasma', 'inferno', 'magma', 'cividis',
             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
             'hot', 'afmhot', 'gist_heat', 'copper']

    xt, yt, zt = task.target_pos
    x0, y0, z0 = task.sim.init_pose[:3]
    
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = plt.subplot(111, projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if len(ep_to_plot) == 1:
        ax.set_title("Trajectory of the trained agent, \n initial: {}, target: {}".\
                 format([int(x0), int(y0), int(z0)], [int(xt), int(yt), int(zt)]))
    else:
        ax.set_title("Trajectories of episode {} to {}, \n initial: {}, target: {}".\
                     format(ep_to_plot[0]+1, ep_to_plot[-1]+1, 
                            [int(x0), int(y0), int(z0)], [int(xt), int(yt), int(zt)]))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_zlim(zmin, zmax)
    for ii, ep in enumerate(ep_to_plot):
        xs = pos_all[ep, :, 0]
        ys = pos_all[ep, :, 1]
        zs = pos_all[ep, :, 2]
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        zs = zs[np.isfinite(zs)]
        ax.plot3D(xs, ys, zs, 'gray') # line
        ax.scatter(xs, ys, zs, c=zs, cmap='GnBu')
    

def plot_test_results(task, results):
    """ 
    plot testing results, one episode
    
    results: output from ddpg_agent.test()
    """
    plt.figure(figsize=(14, 3))
    plt.subplot(151)
    plt.title("target_pos={}".format(task.target_pos))
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    plt.xlabel("time, s")
    
    plt.subplot(152)
    plt.title("velocity, m/s")
    plt.plot(results['time'], results['x_velocity'], label='v_x')
    plt.plot(results['time'], results['y_velocity'], label='v_y')
    plt.plot(results['time'], results['z_velocity'], label='v_z')
    plt.legend()
    plt.xlabel("time, s")
    
    plt.subplot(153)
    plt.title("Euler angles")
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    plt.xlabel("time, s")
    
    plt.subplot(154)
    plt.title("angular velocity")
    plt.plot(results['time'], results['phi_velocity'], label='v_phi')
    plt.plot(results['time'], results['theta_velocity'], label='v_theta')
    plt.plot(results['time'], results['psi_velocity'], label='v_psi')
    plt.legend()
    plt.xlabel("time, s")
    
    plt.subplot(155)
    plt.title("roter speed, rps")
    plt.plot(results['time'], results['rotor_speed1'], label='rotor 1')
    plt.plot(results['time'], results['rotor_speed2'], label='rotor 2')
    plt.plot(results['time'], results['rotor_speed3'], label='rotor 3')
    plt.plot(results['time'], results['rotor_speed4'], label='rotor 4')
    plt.legend()
    plt.xlabel("time, s")
    
    plt.tight_layout()   
    plt.show()