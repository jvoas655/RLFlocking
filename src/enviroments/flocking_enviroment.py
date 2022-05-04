import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

#np.random.seed(0)

def orientation(px, py, qx, qy, rx, ry):
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    return np.where(val > 0, 1, 2)
def onSegment(px, py, qx, qy, rx, ry):
    return np.where((qx <= np.maximum(px, rx)) & (qx >= np.minimum(px, rx)) & (qy <= np.maximum(py, ry)) & (qy >= np.minimum(py, ry)), True, False)
def intersects(point_pairs):
    o1 = orientation(point_pairs[:, 0, 0], point_pairs[:, 0, 1], point_pairs[:, 0, 2], point_pairs[:, 0, 3], point_pairs[:, 1, 0], point_pairs[:, 1, 1])
    o2 = orientation(point_pairs[:, 0, 0], point_pairs[:, 0, 1], point_pairs[:, 0, 2], point_pairs[:, 0, 3], point_pairs[:, 1, 2], point_pairs[:, 1, 3])
    o3 = orientation(point_pairs[:, 1, 0], point_pairs[:, 1, 1], point_pairs[:, 1, 2], point_pairs[:, 1, 3], point_pairs[:, 0, 0], point_pairs[:, 0, 1])
    o4 = orientation(point_pairs[:, 1, 0], point_pairs[:, 1, 1], point_pairs[:, 1, 2], point_pairs[:, 1, 3], point_pairs[:, 0, 2], point_pairs[:, 0, 3])
    gc = np.where((o1 != o2) & (o3 != o4), True, False)
    # Does not handle special cases for now. Causing zero impacts
    #sp1 = np.where((o1 == 0) & onSegment(point_pairs[:, 0, 0], point_pairs[:, 0, 1], point_pairs[:, 1, 0], point_pairs[:, 1, 1], point_pairs[:, 0, 2], point_pairs[:, 0, 3]), True, False)
    #sp2 = np.where((o2 == 0) & onSegment(point_pairs[:, 0, 0], point_pairs[:, 0, 1], point_pairs[:, 1, 2], point_pairs[:, 1, 3], point_pairs[:, 0, 2], point_pairs[:, 0, 3]), True, False)
    #sp3 = np.where((o3 == 0) & onSegment(point_pairs[:, 1, 0], point_pairs[:, 1, 1], point_pairs[:, 0, 0], point_pairs[:, 0, 1], point_pairs[:, 1, 2], point_pairs[:, 1, 3]), True, False)
    #sp4 = np.where((o4 == 0) & onSegment(point_pairs[:, 1, 0], point_pairs[:, 1, 1], point_pairs[:, 0, 2], point_pairs[:, 0, 3], point_pairs[:, 1, 2], point_pairs[:, 1, 3]), True, False)
    return np.where(gc, True, False)

def shift(A, dx, dy):
    new_A = np.roll(A, (dx, dy), (0, 1))
    if (dx > 0):
        new_A[:dx, :, :] = 0
    elif (dx < 0):
        new_A[dx:, :, :] = 0
    if (dy > 0):
        new_A[:, :dy, :] = 0
    elif (dy < 0):
        new_A[:, dy:, :] = 0
    return new_A

class FlockEnviroment:
    def __init__(self, num_agents, physics_params = None):

        self.num_agents = num_agents # Determines number of agents in flock
        self.airflow_bin_size = 0.02 # Determines bin size
        self.airflow_velocity_transfer = 1.0 # Determines ratio of velocity transfered to local bin air flow
        self.airflow_conv_decay = 0.9 # Determines air flow spread factor
        self.velocity_time_decay = 0.5 # Determines time decay of velocity
        self.airflow_time_decay = 0.8 # Determines time decay of air flow
        self.airflow_conv_spread = 1 # Determines air flow spread range
        self.impact_energy_cost = 0.10 # Energy cost per impact
        self.impact_velocity_decay = 0.9 # How much velocity is lost on collisions
        self.neighbor_view_count = 10 # Number of nearest neighbors visible
        self.base_energy_cost = 0.01 # Base energy that is taken to move at each time time_step
        self.airflow_assist_factor = 0.9 # How much airflow can increase/decrease energy usage
        self.maximum_velocity = 0.05
        self.acceleration_scale = 0.1
        self.dimensions = 2 # Keep at 2 for now


        """"""
        self.with_boundary_observation = False 

        if (physics_params):
            if ("airflow_bin_size" in physics_params):
                self.airflow_bin_size = physics_params["airflow_bin_size"]
            if ("airflow_velocity_transfer" in physics_params):
                self.airflow_velocity_transfer = physics_params["airflow_velocity_transfer"]
            if ("airflow_conv_decay" in physics_params):
                self.airflow_conv_decay = physics_params["airflow_conv_decay"]
            if ("velocity_time_decay" in physics_params):
                self.velocity_time_decay = physics_params["velocity_time_decay"]
            if ("airflow_time_decay" in physics_params):
                self.airflow_time_decay = physics_params["airflow_time_decay"]
            if ("airflow_conv_spread" in physics_params):
                self.airflow_conv_spread = physics_params["airflow_conv_spread"]
            if ("impact_energy_cost" in physics_params):
                self.impact_energy_cost = physics_params["impact_energy_cost"]
            if ("impact_velocity_decay" in physics_params):
                self.impact_velocity_decay = physics_params["impact_velocity_decay"]
            if ("neighbor_view_count" in physics_params):
                self.neighbor_view_count = physics_params["neighbor_view_count"]
            if ("base_energy_cost" in physics_params):
                self.base_energy_cost = physics_params["base_energy_cost"]
            if ("airflow_assist_factor" in physics_params):
                self.airflow_assist_factor = physics_params["airflow_assist_factor"]
        # Get bin count based on bin size
        self.airflow_bin_count = int(1 / self.airflow_bin_size)
        if (self.airflow_conv_spread  > self.airflow_bin_count):
            print("Airflow bin count ({count:d}) lower then airflow conv spread ({conv:d}). Consider increasing airflow bin size ({size:f})".format(count = self.airflow_bin_count, conv=self.airflow_conv_spread, size = self.airflow_bin_size))
        if (self.neighbor_view_count > self.num_agents - 1):
            print("State view ({view:d}) is larger then the number of other agents ({agents:d})".format(view = self.neighbor_view_count, agents = self.num_agents - 1))

    def get_observation_size(self):
        if self.with_boundary_observation:
            return ((3 + self.neighbor_view_count * 2) * self.dimensions) + 2 * self.dimensions - 1
        else:
            return ((3 + self.neighbor_view_count * 2) * self.dimensions) -1
        

    def reset(self):
        # Initialize agent properties
        self.agent_positions = np.random.random((self.num_agents, self.dimensions))
        self.position_kd_tree = KDTree(self.agent_positions, leafsize = self.neighbor_view_count)
        self.agent_velocities = np.zeros((self.num_agents, self.dimensions))
        self.agent_accelerations = np.zeros((self.num_agents, self.dimensions))
        self.agent_energies = np.ones(self.num_agents)
        # Initialize values for distance rewards
        self.agent_travel_distances = np.zeros(self.num_agents)
        self.time_step = 0
        # Initialize air flow arrays
        self.airflow_store = np.zeros((self.airflow_bin_count, self.airflow_bin_count, self.dimensions))
        self.airflow_swap = np.zeros((self.airflow_bin_count, self.airflow_bin_count, self.dimensions))
        self.agent_airflows = np.zeros((self.num_agents, self.dimensions))
        self.agent_airflow_incidence_angle = np.zeros(self.num_agents)
        x_grid, y_grid =  np.meshgrid(np.arange(-1 * self.airflow_conv_spread, self.airflow_conv_spread + 1), np.arange(-1 * self.airflow_conv_spread, self.airflow_conv_spread + 1))
        self.airflow_conv_decay_grid = self.airflow_conv_decay ** np.sqrt(x_grid ** 2 + y_grid ** 2).reshape(self.dimensions * self.airflow_conv_spread + 1, -1)
        self.airflow_conv_decay_grid_sum = np.sum(self.airflow_conv_decay_grid) # This may be wrong, may need to be by location
        # Initalize arrays for collisions
        self.agent_step_segments = np.zeros((self.num_agents,self.dimensions * 2))
        self.agent_step_segments[:, self.dimensions:self.dimensions * 2] = self.agent_positions
        self.collision_agent_pairs = np.vstack(np.meshgrid(np.arange(self.num_agents), np.arange(self.num_agents))).reshape(self.dimensions, self.num_agents**2).T
        self.collision_agent_pairs = np.squeeze(self.collision_agent_pairs[np.where(self.collision_agent_pairs[:, 0] > self.collision_agent_pairs[:, 1]), :])
        self.episode_collisions = 0
        # Initialize arrays for post episode display
        self.airflow_sequence_store = [self.airflow_store]
        self.posiiton_sequence_store = [self.agent_positions]

        self.calc_states()

        return self.observation_collection


    def calc_states(self):
        # Clear prior observations
        self.observation_collection = np.zeros((self.num_agents, self.get_observation_size()))
        self.reward_collection = np.zeros(self.num_agents)

        # Get which airflow bin agents are in for this time step
        self.timestep_agent_bin_locations = (self.agent_positions // self.airflow_bin_size).astype("int")
        # Get nearest neighbors for this timestep
        timestep_neighbors = self.position_kd_tree.query(self.agent_positions, k = self.neighbor_view_count + 1)[1][:, 1:]
        y_align_rotations = np.zeros((self.num_agents, self.dimensions, self.dimensions))
        y_align_rotations[:, 0, :] = self.agent_velocities
        y_align_rotations[:, 1, 0] = -1 * self.agent_velocities[:, 1]
        y_align_rotations[:, 1, 1] = self.agent_velocities[:, 0]
        y_align_rotation_norms2 = np.sqrt(np.sum(self.agent_velocities**2, axis = 1))
        self.inverse_rotations = np.zeros((self.num_agents, self.dimensions, self.dimensions))

        for agent_ind in range(self.num_agents):
            # Get air flow at agents binned location
            self.agent_airflows[agent_ind, :] = self.airflow_store[self.timestep_agent_bin_locations[agent_ind, 0], self.timestep_agent_bin_locations[agent_ind, 1], :]
            # If agent is out of energy it can not accelerate
            if (self.agent_energies[agent_ind] > 0):
                # Setup state value to obtain acceleration
                # First two states are agent position
                # Second two states are agent velocity. This state and all following states are rotated to match the agents velocity to [1, 0] for symmetry
                # Third two states are airflow at agent location
                # Next batch of states is the reletive positions of all nearest agents
                # Final batch of states is the velocity of all nearest agents
                timestep_state = np.zeros(self.get_observation_size() + 1).reshape(-1, self.dimensions)
                state_ind = 0


                if (y_align_rotation_norms2[agent_ind] != 0):
                    y_align_agent_rotation = y_align_rotations[agent_ind, :, :] / y_align_rotation_norms2[agent_ind]
                    self.inverse_rotations[agent_ind] = np.linalg.inv(y_align_agent_rotation)
                else:
                    y_align_agent_rotation = np.identity(self.dimensions)
                    self.inverse_rotations[agent_ind] = np.identity(self.dimensions)
                timestep_state[state_ind, :] = np.flip(np.matmul(y_align_agent_rotation, self.agent_velocities[agent_ind]))
                state_ind += 1
                timestep_state[state_ind, :] = self.agent_positions[agent_ind]
                state_ind += 1
                timestep_state[state_ind, :] = np.matmul(y_align_agent_rotation, self.agent_airflows[agent_ind, :])
                state_ind += 1
                
                """ Add distance to box boundaries
                """
                if self.with_boundary_observation:
                    for d in range(self.dimensions):
                        timestep_state[state_ind, :] = self.agent_positions[agent_ind][d] - 0.
                        state_ind += 1
                        timestep_state[state_ind, :] = self.agent_positions[agent_ind][d] - 1.
                        state_ind += 1



                for neighbor in range(self.neighbor_view_count):
                    timestep_state[state_ind, :] = np.matmul(y_align_agent_rotation, ((self.agent_positions[timestep_neighbors[agent_ind, neighbor]] - self.agent_positions[agent_ind])))
                    state_ind += 1
                    timestep_state[state_ind, :] = np.matmul(y_align_agent_rotation, self.agent_velocities[timestep_neighbors[agent_ind, neighbor]])
                    state_ind += 1
                timestep_state = timestep_state.flatten()[1:] # Throw away zero velocity since its always rotated to be along [1, 0]
                '''
                timestep_state[state_ind, :] = self.agent_velocities[agent_ind]
                state_ind += 1
                timestep_state[state_ind, :] = self.agent_positions[agent_ind] - 0.5
                state_ind += 1
                timestep_state[state_ind, :] = self.agent_airflows[agent_ind, :]
                state_ind += 1
                for neighbor in range(self.neighbor_view_count):
                    timestep_state[state_ind, :] = ((self.agent_positions[timestep_neighbors[agent_ind, neighbor]] - self.agent_positions[agent_ind]))
                    state_ind += 1
                    timestep_state[state_ind, :] = self.agent_velocities[timestep_neighbors[agent_ind, neighbor]]
                    state_ind += 1
                timestep_state = timestep_state.flatten() # Throw away zero velocity since its always rotated to be along [1, 0]
                '''
                self.observation_collection[agent_ind, :] = timestep_state

    def step(self, actions):
        # Assume zero acceleration
        self.agent_accelerations[:, :] = 0
        for agent_ind in range(self.num_agents):
            if (self.agent_energies[agent_ind] > 0):
                self.agent_accelerations[agent_ind, :] = np.matmul(self.inverse_rotations[agent_ind], actions[agent_ind, :])
                #self.agent_accelerations[agent_ind, :] = actions[agent_ind, :]
        self.agent_accelerations *= self.acceleration_scale
                # Should query model for acceleration decision and apply any limits here. May need to apply rotation afterwards here
                # Pseudo random acceleration for now, with a bias towards center
                #self.agent_accelerations[agent_ind, :] = (np.random.random(self.dimensions) - 0.5) * 0.01 + -1 * (self.agent_positions[agent_ind, :] - 0.5) * np.random.random(1) * 0.002
        # Get effective airflow at location reletive to agent velocity
        airflow_velocity_difference = self.agent_airflows - self.agent_velocities
        # Get angle between effective airflow and desired acceleration
        avd_acceleraton_dot_norm = np.multiply(np.linalg.norm(airflow_velocity_difference, axis = 1), np.linalg.norm(self.agent_accelerations, axis = 1))
        self.agent_airflow_incidence_angle[:] = 0
        for agent_ind in range(self.num_agents):
            if (avd_acceleraton_dot_norm[agent_ind] != 0):
                self.agent_airflow_incidence_angle[agent_ind] = np.dot(airflow_velocity_difference[agent_ind, :], self.agent_accelerations[agent_ind, :]) / avd_acceleraton_dot_norm[agent_ind]
        # Deduct modified energy cost
        self.agent_energies -= (self.base_energy_cost - (self.base_energy_cost * self.airflow_assist_factor * self.agent_airflow_incidence_angle))
        # Apply flat decay to velocity and add acceleration
        self.agent_velocities = np.clip(self.velocity_time_decay * self.agent_velocities + self.agent_accelerations, -1 * self.maximum_velocity, self.maximum_velocity)
        # Apply flat decay to entire airflow state
        self.airflow_store = self.airflow_time_decay * self.airflow_store
        # Transfer velocities from this timestep into the airflow approximation grid
        for agent_ind in range(self.num_agents):
            self.airflow_store[self.timestep_agent_bin_locations[agent_ind, 0], self.timestep_agent_bin_locations[agent_ind, 1], :] += self.airflow_velocity_transfer * self.agent_velocities[agent_ind, :]
        # Update agent positions and travel distances, and update KDTree
        self.agent_positions += self.agent_velocities
        self.position_kd_tree = KDTree(self.agent_positions, leafsize = self.neighbor_view_count)
        potential_pairs = self.position_kd_tree.query_pairs(2 * self.maximum_velocity, output_type="ndarray")
        self.agent_travel_distances += np.linalg.norm(self.agent_velocities, axis = 1)

        # Detect collisions with walls in this timestep. Decay and invert velocity in dimensions with a impact
        timestep_agent_collisions = np.zeros(self.num_agents)
        for dimension in range(self.dimensions):
            timestep_wall_hits = np.where((self.agent_positions[:, dimension] > 1) | (self.agent_positions[:, dimension] < 0), True, False)
            self.agent_velocities[:, dimension] = np.where(timestep_wall_hits, -1 * self.impact_velocity_decay * self.agent_velocities[:, dimension], self.agent_velocities[:, dimension])
            timestep_agent_collisions += np.where(timestep_wall_hits, 1, 0)
        # Clip all positions without [0,1] bounds
        self.agent_positions = np.clip(self.agent_positions, 0, 1)
        self.position_kd_tree = KDTree(self.agent_positions, leafsize = self.neighbor_view_count)
        # Detect collisions between agents during this time step. No position/velocity modification applied
        self.agent_step_segments = np.roll(self.agent_step_segments, self.dimensions, axis = 1)
        self.agent_step_segments[:, self.dimensions:self.dimensions * 2] = self.agent_positions
        timestep_position_segment_pairs = self.agent_step_segments[potential_pairs]
        timestep_position_intersects = intersects(timestep_position_segment_pairs)
        timestep_agent_collisions += np.bincount(potential_pairs[timestep_position_intersects].reshape(-1,), minlength=self.num_agents)
        self.reward_collection = -1 * timestep_agent_collisions

        # self.reward_collection = self.agent_travel_distances

        # Subtract energy based on number of collisions (agent-agent, agent-wall) in this timestep
        self.agent_energies -= timestep_agent_collisions * self.impact_energy_cost
        # Perform convolution in swap array to spread airflow for this time step
        for dx in range(-1 * self.airflow_conv_spread, self.airflow_conv_spread + 1):
            for dy in range(-1 * self.airflow_conv_spread, self.airflow_conv_spread + 1):
                self.airflow_swap += self.airflow_conv_decay_grid[dx + self.airflow_conv_spread, dy + self.airflow_conv_spread] * shift(self.airflow_store, dx, dy)
        # Normalize airflow from convolution and move into store array. Clear swap for next timestep
        self.airflow_store = self.airflow_swap / self.airflow_conv_decay_grid_sum
        self.airflow_swap[:, :, :] = 0
        # Save airflow and positions for display
        self.airflow_sequence_store.append(self.airflow_store)
        self.posiiton_sequence_store.append(self.agent_positions)

        # Make a copy of observations, actions, and rewards
        observation_collection_clone = self.observation_collection.copy()
        reward_collection_clone = self.reward_collection.copy()
        self.episode_collisions += np.sum(timestep_agent_collisions)
        # Clear collections and calculate next states
        self.calc_states()

        # Increment time step and return True if episode is still going
        self.time_step += 1
        return observation_collection_clone, reward_collection_clone, self.observation_collection, not np.any(self.agent_energies > 0)
    def display_last_episode(self):
        X, Y = np.meshgrid(np.arange(0, self.airflow_bin_count), np.arange(0, self.airflow_bin_count))
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.set(adjustable='box', aspect='equal')
        Q = axes[0].quiver(X * self.airflow_bin_size, Y * self.airflow_bin_size, self.airflow_sequence_store[-1][:, :, 0].flatten(), self.airflow_sequence_store[-1][:, :, 1].flatten(), pivot='mid', color='r', units='inches', scale=0.002)
        axes[0].set_title("Air Flow")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        L = axes[1].plot(self.posiiton_sequence_store[0][:, 0], self.posiiton_sequence_store[0][:, 1], 'o', 'black')[0]
        axes[1].set_title("Agent Positions (Random Acceleration with Regularizer Towards the Origin)")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        def update(i, L, Q):
             Q.set_UVC(self.airflow_sequence_store[i][:, :, 0].flatten(), self.airflow_sequence_store[i][:, :, 1].flatten())
             L.set_data(self.posiiton_sequence_store[i][:, 0], self.posiiton_sequence_store[i][:, 1])
             return (Q, L)
        animator = animation.FuncAnimation(fig, update, fargs=(L, Q), frames = range(self.time_step), interval = 200, blit = False)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    st = time.time()
    num_episodes = 1
    f = FlockEnviroment(200)
    for episode in range(num_episodes):
        f.reset()
        while (not f.step()[-1]):
            continue
    print(time.time() - st)
    f.display_last_episode()
