import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import animation, patches
from matplotlib.animation import FuncAnimation
import matplotlib.collections as mcoll

# Project includes.
from torch_robotics.torch_utils.torch_utils import to_numpy
from mrrd.plotting.base import remove_borders, remove_axes_labels_ticks


def create_fig_and_axes(dim=2):
    fig = plt.figure(layout='tight')
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    return fig, ax


def _to_numpy_list(data):
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        return [t.detach().cpu().numpy()[:2] if torch.is_tensor(t) and t.ndim == 1 else (t.detach().cpu().numpy()[:, :2] if torch.is_tensor(t) else np.asarray(t)[..., :2]) for t in data]
    arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    if arr.ndim == 3:
        return [arr[i, :, :2] for i in range(arr.shape[0])]
    elif arr.ndim == 2:
        return [arr[:, :2]]
    return []


def _to_positions_list(positions):
    if positions is None:
        return []
    if isinstance(positions, (list, tuple)):
        return [p.detach().cpu().numpy()[:2] if torch.is_tensor(p) else np.asarray(p)[:2] for p in positions]
    arr = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
    if arr.ndim == 2:
        return [arr[i, :2] for i in range(arr.shape[0])]
    elif arr.ndim == 1:
        return [arr[:2]]
    return []


class PlanningVisualizer:

    def __init__(self, task=None, planner=None):
        self.task = task
        self.env = self.task.env
        self.robot = self.task.robot
        self.planner = planner

        # self.colors = {'collision': 'black', 'free': 'cornsilk'}
        self.colors = {'collision': 'black', 'free': 'lightcyan'}
        self.colors_robot = {'collision': 'black', 'free': 'darkorange'}
        self.cmaps = {'collision': 'Greys', 'free': 'Oranges'}
        self.cmaps_robot = {'collision': 'Greys', 'free': 'YlOrRd'}

    def render_robot_trajectories(self, fig=None, ax=None, render_planner=False, trajs=None, traj_best=None,
                                  show_robot_in_image=False, constraints_l=None, **kwargs):
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        if render_planner:
            self.planner.render(ax)
        self.env.render(ax)
        if trajs is not None:
            _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.task.get_trajs_collision_and_free(trajs, return_indices=True)
            kwargs['colors'] = []
            for i in range(len(trajs_coll_idxs) + len(trajs_free_idxs)):
                kwargs['colors'].append(self.colors['collision'] if i in trajs_coll_idxs else self.colors['free'])
        self.robot.render_trajectories(ax, trajs=trajs, constraints_l=constraints_l, **kwargs)
        if traj_best is not None:
            kwargs['colors'] = ['blue']
            self.robot.render_trajectories(ax, trajs=traj_best.unsqueeze(0), constraints_l=constraints_l, **kwargs)

        # Add the robot with tail at the third-way time step.
        if show_robot_in_image:
            t_mid = trajs.shape[1] // 5
            qs = trajs[:, t_mid, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            for q in qs:
                q_tail = []
                tail_length = min(5, t_mid)
                for di in range(tail_length):
                    q_tail.append(trajs[:, t_mid - di, :].squeeze())
                kwargs['q_tail'] = q_tail
                # Random color from colormap for the robot and tail. From cmap tab20(do c optionally).
                color = plt.cm.tab20(np.random.randint(0, 20))
                self.robot.render(
                    ax, q=q,
                    color=color,
                    arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                    cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps['free'],
                    **kwargs
                )

        return fig, ax

    def animate_robot_trajectories(
            self, trajs=None, start_state=None, goal_state=None, waypoints=None,
            plot_trajs=False,
            n_frames=10,
            constraints=None,
            **kwargs
    ):
        if trajs is None:
            return

        assert trajs.ndim == 3
        B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_selection = trajs[:, idxs, :]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            """
            Draw all the robots at step i for all trajectories.
            Also add a robot configuration at any constraints.
            """
            print(idxs[i], "/", H, end="\r")
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H-1}")
            if plot_trajs:
                self.render_robot_trajectories(
                    fig=fig, ax=ax, trajs=trajs, start_state=start_state, goal_state=goal_state, **kwargs
                )
            else:
                self.env.render(ax)

            # TODO - implement batched version
            qs = trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            for i, q in enumerate(qs):
                self.robot.render(
                    ax, q=q,
                    color=self.colors_robot['collision'] if self.task.compute_collision(q, margin=0.0) else self.colors_robot['free'],
                    arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                    cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps['free'],
                    **kwargs
                )

            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')
            if waypoints is not None:
                for wp in waypoints:
                    self.robot.render(ax, wp, color='orange', cmap='Oranges')
            step = idxs[i]
            if constraints is not None:
                for constraint in constraints:
                    # For Constraint object passed.
                    if constraint.get_t_range()[0] <= step <= constraint.get_t_range()[1]:
                        self.robot.render(ax, constraint.get_q(), color='red', cmap='Reds')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_multi_robot_trajectories(
            self, trajs_l=None, start_state_l=None, goal_state_l=None,
            plot_trajs=False,
            n_frames=10,
            constraints=None,
            colors=None,
            **kwargs
    ):
        assert len(colors) == len(trajs_l)

        assert trajs_l[0].ndim == 3
        B, H, D = trajs_l[0].shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_l_selection = []
        for trajs in trajs_l:
            trajs_selection = trajs[:, idxs, :]
            trajs_l_selection.append(trajs_selection)

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            """
            Draw all the robots at step i for all trajectories.
            Also add a robot configuration at any constraints.
            """
            print(idxs[i], "/", H, end="\r")
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H-1}")
            if plot_trajs:
                for trajs, color, start_state, goal_state in zip(trajs_l_selection, colors, start_state_l, goal_state_l):
                    self.render_robot_trajectories(
                        fig=fig,
                        ax=ax,
                        trajs=trajs,
                        start_state=start_state,
                        goal_state=goal_state,
                        colors=[color]*len(trajs),
                        **kwargs
                    )
            else:
                self.env.render(ax)

            for trajs_selection, color, start_state, goal_state in zip(trajs_l_selection, colors, start_state_l, goal_state_l):
                qs = trajs_selection[:, i, :]  # batch, q_dim
                if qs.ndim == 1:
                    qs = qs.unsqueeze(0)  # interface (batch, q_dim)
                for q in qs:
                    q_tail = []
                    for di in range(5):
                        q_tail.append(trajs_selection[:, i - di, :].squeeze())
                    kwargs['q_tail'] = q_tail
                    self.robot.render(
                        ax, q=q,
                        color=color,
                        arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                        cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps['free'],
                        **kwargs
                    )

                # if start_state is not None:
                #     self.robot.render(ax, start_state, color='green', cmap='Greens')
                # if goal_state is not None:
                #     self.robot.render(ax, goal_state, color='purple', cmap='Purples')
                step = idxs[i]
                if constraints is not None:
                    for constraint in constraints:
                        if constraint.get_t_range()[0] <= step <= constraint.get_t_range()[1]:
                            self.robot.render(ax, constraint.get_q(), color='red', cmap='Reds')

                remove_borders(ax)
                remove_axes_labels_ticks(ax)
                ax.set_title('')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_opt_iters_robots(
            self, trajs=None, traj_best=None, start_state=None, goal_state=None,
            n_frames=10,
            **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            self.render_robot_trajectories(
                fig=fig, ax=ax, trajs=trajs_selection[i],
                traj_best=traj_best if i == n_frames - 1 else None,
                start_state=start_state, goal_state=goal_state, **kwargs
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def plot_joint_space_state_trajectories(
            self,
            fig=None, axs=None,
            trajs=None,
            traj_best=None,
            pos_start_state=None, pos_goal_state=None,
            vel_start_state=None, vel_goal_state=None,
            set_joint_limits=True,
            **kwargs
    ):
        if trajs is None:
            return
        trajs_np = to_numpy(trajs)

        assert trajs_np.ndim == 3
        B, H, D = trajs_np.shape

        # Separate trajectories in collision and free (not in collision)
        trajs_coll, trajs_free = self.task.get_trajs_collision_and_free(trajs)

        trajs_coll_pos_np = to_numpy([])
        trajs_coll_vel_np = to_numpy([])
        if trajs_coll is not None:
            trajs_coll_pos_np = to_numpy(self.robot.get_position(trajs_coll))
            trajs_coll_vel_np = to_numpy(self.robot.get_velocity(trajs_coll))

        trajs_free_pos_np = to_numpy([])
        trajs_free_vel_np = to_numpy([])
        if trajs_free is not None:
            trajs_free_pos_np = to_numpy(self.robot.get_position(trajs_free))
            trajs_free_vel_np = to_numpy(self.robot.get_velocity(trajs_free))

        if pos_start_state is not None:
            pos_start_state = to_numpy(pos_start_state)
        if vel_start_state is not None:
            vel_start_state = to_numpy(vel_start_state)
        if pos_goal_state is not None:
            pos_goal_state = to_numpy(pos_goal_state)
        if vel_goal_state is not None:
            vel_goal_state = to_numpy(vel_goal_state)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 2, squeeze=False)
        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[-1, 0].set_xlabel('Timesteps')
        axs[-1, 1].set_xlabel('Timesteps')
        timesteps = np.arange(H).reshape(1, -1)
        for i, ax in enumerate(axs):
            for trajs_filtered, color in zip([(trajs_coll_pos_np, trajs_coll_vel_np), (trajs_free_pos_np, trajs_free_vel_np)],
                                             ['black', 'orange']):
                # Positions and velocities
                for j, trajs_filtered_ in enumerate(trajs_filtered):
                    if trajs_filtered_.size > 0:
                        timesteps_ = np.repeat(timesteps, trajs_filtered_.shape[0], axis=0)
                        plot_multiline(ax[j], timesteps_, trajs_filtered_[..., i], color=color, **kwargs)

            if traj_best is not None:
                traj_best_pos = self.robot.get_position(traj_best)
                traj_best_vel = self.robot.get_velocity(traj_best)
                traj_best_pos_np = to_numpy(traj_best_pos)
                traj_best_vel_np = to_numpy(traj_best_vel)
                plot_multiline(ax[0], timesteps, traj_best_pos_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[1], timesteps, traj_best_vel_np[..., i].reshape(1, -1), color='blue', **kwargs)

            # Start and goal
            if pos_start_state is not None:
                ax[0].scatter(0, pos_start_state[i], color='green')
            if vel_start_state is not None:
                ax[1].scatter(0, vel_start_state[i], color='green')
            if pos_goal_state is not None:
                ax[0].scatter(H-1, pos_goal_state[i], color='purple')
            if vel_goal_state is not None:
                ax[1].scatter(H-1, vel_goal_state[i], color='purple')
            # Y label
            ax[0].set_ylabel(f'q_{i}')
            # Set limits
            if set_joint_limits:
                ax[0].set_ylim(self.robot.q_min_np[i], self.robot.q_max_np[i])
                # ax[1].set_ylim(self.robot.q_vel_min_np[i], self.robot.q_vel_max_np[i])

        return fig, axs

    def animate_opt_iters_joint_space_state(
            self, trajs=None, traj_best=None, n_frames=10, **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, axs = self.plot_joint_space_state_trajectories(trajs=trajs_selection[0], **kwargs)

        def animate_fn(i):
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S-1}")
            self.plot_joint_space_state_trajectories(
                fig=fig, axs=axs,
                trajs=trajs_selection[i], **kwargs
            )
            if i == n_frames -1 and traj_best is not None:
                self.plot_joint_space_state_trajectories(
                    fig=fig, axs=axs,
                    trajs=trajs_selection[i],
                    traj_best=traj_best, **kwargs
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def render_ego_centric_view(
            self, robot_trajs=None, human_trajs=None, animation_duration: float = 5.0,
            video_filepath: str = None, robot_radius: float = 0.15, human_radius: float = 0.3,
            window_size: float = 6.0, waypoint_rate: int = 15, start_positions=None, goal_positions=None
    ):
        robot_trajs_list = _to_numpy_list(robot_trajs)
        human_trajs_np = (
            human_trajs.detach().cpu().numpy() if torch.is_tensor(human_trajs)
            else (np.asarray(human_trajs) if human_trajs is not None else np.zeros((0, 0, 2)))
        )

        B = len(robot_trajs_list)
        if B == 0:
            print("No robot trajectories to render.")
            return

        N_humans = human_trajs_np.shape[0] if human_trajs_np.ndim == 3 else 0
        H_human = human_trajs_np.shape[1] if N_humans > 0 else 0

        start_pos_list = _to_positions_list(start_positions)
        goal_pos_list = _to_positions_list(goal_positions)

        cmap = plt.get_cmap('tab10')
        base_path = os.path.splitext(video_filepath)[0] if video_filepath else "ego"

        for main_robot_idx in range(B):
            ego_robot_traj = robot_trajs_list[main_robot_idx]
            H_robot = ego_robot_traj.shape[0]
            max_other_len = max(t.shape[0] for t in robot_trajs_list)
            frames = max(H_robot, H_human, max_other_len)

            curr_video_path = f"{base_path}_robot_{main_robot_idx}.gif"

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)

            half_win = window_size / 2.0
            ax.set_xlim(-half_win, half_win)
            ax.set_ylim(-half_win, half_win)
            ax.set_title(f"Ego-Centric View (Robot {main_robot_idx})")

            # Main robot artists
            main_color = cmap(main_robot_idx % 10)
            main_line, = ax.plot([], [], color=main_color, alpha=0.5, linewidth=2, label='Ego Path')
            main_wp, = ax.plot([], [], linestyle='None', marker='o', color=main_color, markersize=4, alpha=0.8)
            main_circle = patches.Circle((0, 0), radius=robot_radius, color=main_color, alpha=0.9, zorder=10)
            ax.add_patch(main_circle)

            main_start, = ax.plot([], [], marker='s', color=main_color, markersize=10, linestyle='None', alpha=0.8, zorder=8, label='Start')
            main_goal, = ax.plot([], [], marker='*', color=main_color, markersize=14, linestyle='None', alpha=0.8, zorder=8, label='Goal')

            # Other robots
            other_robots = []
            for b_idx in range(B):
                if b_idx == main_robot_idx:
                    continue
                color = cmap(b_idx % 10)
                l, = ax.plot([], [], color=color, alpha=0.4, linewidth=2)
                c = patches.Circle((999, 999), radius=robot_radius, color=color, alpha=0.8, zorder=5)
                ax.add_patch(c)
                s_m, = ax.plot([], [], marker='s', color=color, markersize=8, linestyle='None', alpha=0.6, zorder=4)
                g_m, = ax.plot([], [], marker='*', color=color, markersize=12, linestyle='None', alpha=0.6, zorder=4)
                other_robots.append({'idx': b_idx, 'line': l, 'circle': c, 'start': s_m, 'goal': g_m})

            # Humans
            human_artists = []
            for h in range(N_humans):
                hl, = ax.plot([], [], color='grey', alpha=0.4, linewidth=2)
                hc = patches.Circle((999, 999), radius=human_radius, color='grey', alpha=0.6, zorder=5)
                ax.add_patch(hc)
                human_artists.append({'idx': h, 'line': hl, 'circle': hc})

            def init():
                main_circle.center = (0, 0)
                main_line.set_data([], [])
                main_wp.set_data([], [])
                main_start.set_data([], [])
                main_goal.set_data([], [])
                artists = [main_circle, main_line, main_wp, main_start, main_goal]
                for item in other_robots:
                    item['circle'].center = (999, 999)
                    item['line'].set_data([], [])
                    item['start'].set_data([], [])
                    item['goal'].set_data([], [])
                    artists.extend([item['circle'], item['line'], item['start'], item['goal']])
                for item in human_artists:
                    item['circle'].center = (999, 999)
                    item['line'].set_data([], [])
                    artists.extend([item['circle'], item['line']])
                return artists

            def update(frame):
                curr_pos = ego_robot_traj[min(frame, H_robot - 1)]
                rel_history = ego_robot_traj[:min(frame, H_robot - 1) + 1] - curr_pos
                main_line.set_data(rel_history[:, 0], rel_history[:, 1])

                if waypoint_rate > 0:
                    main_wp.set_data(rel_history[::waypoint_rate, 0], rel_history[::waypoint_rate, 1])

                if main_robot_idx < len(start_pos_list):
                    rel_s = start_pos_list[main_robot_idx] - curr_pos
                    main_start.set_data([rel_s[0]], [rel_s[1]])
                else:
                    main_start.set_data([], [])

                if main_robot_idx < len(goal_pos_list):
                    rel_g = goal_pos_list[main_robot_idx] - curr_pos
                    main_goal.set_data([rel_g[0]], [rel_g[1]])
                else:
                    main_goal.set_data([], [])

                artists = [main_circle, main_line, main_wp, main_start, main_goal]

                for item in other_robots:
                    traj = robot_trajs_list[item['idx']]
                    o_idx = min(frame, traj.shape[0] - 1)
                    rel_o = traj[o_idx] - curr_pos
                    item['circle'].center = (rel_o[0], rel_o[1])

                    rel_o_hist = traj[:o_idx + 1] - curr_pos
                    item['line'].set_data(rel_o_hist[:, 0], rel_o_hist[:, 1])

                    if item['idx'] < len(start_pos_list):
                        rel_s = start_pos_list[item['idx']] - curr_pos
                        item['start'].set_data([rel_s[0]], [rel_s[1]])
                    else:
                        item['start'].set_data([], [])

                    if item['idx'] < len(goal_pos_list):
                        rel_g = goal_pos_list[item['idx']] - curr_pos
                        item['goal'].set_data([rel_g[0]], [rel_g[1]])
                    else:
                        item['goal'].set_data([], [])

                    artists.extend([item['circle'], item['line'], item['start'], item['goal']])

                for item in human_artists:
                    h_idx = min(frame, H_human - 1)
                    rel_h = human_trajs_np[item['idx'], h_idx] - curr_pos
                    item['circle'].center = (rel_h[0], rel_h[1])
                    rel_h_hist = human_trajs_np[item['idx'], :h_idx + 1] - curr_pos
                    item['line'].set_data(rel_h_hist[:, 0], rel_h_hist[:, 1])
                    artists.extend([item['circle'], item['line']])

                return artists

            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
            fps = max(1, int(frames / animation_duration))
            ani.save(curr_video_path, writer=animation.PillowWriter(fps=fps))
            plt.close(fig)
            print(f'Saved ego-centric gif for robot {main_robot_idx} to {curr_video_path}')

    def render_recent_result_with_humans(
            self, robot_trajs=None, human_trajs=None, animation_duration: float = 5.0,
            video_filepath: str = None, robot_radius: float = 0.15, human_radius: float = 0.3,
            start_positions=None, goal_positions=None
    ):
        robot_trajs_list = _to_numpy_list(robot_trajs)
        human_trajs_np = (
            human_trajs.detach().cpu().numpy() if torch.is_tensor(human_trajs)
            else (np.asarray(human_trajs) if human_trajs is not None else np.zeros((0, 0, 2)))
        )

        B = len(robot_trajs_list)
        N_humans = human_trajs_np.shape[0] if human_trajs_np.ndim == 3 else 0
        if B == 0 and N_humans == 0:
            print("No trajectories to render.")
            return

        start_pos_list = _to_positions_list(start_positions)
        goal_pos_list = _to_positions_list(goal_positions)

        max_robot_len = max(t.shape[0] for t in robot_trajs_list) if robot_trajs_list else 0
        H_human = human_trajs_np.shape[1] if N_humans > 0 else 0
        frames = max(max_robot_len, H_human)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('on')
        self.env.render(ax)

        human_color = 'grey'
        human_lines = [ax.plot([], [], color=human_color, alpha=0.4, linewidth=2)[0] for _ in range(N_humans)]
        human_circles = [patches.Circle((999, 999), radius=human_radius, color=human_color, alpha=0.6) for _ in range(N_humans)]
        for hc in human_circles:
            ax.add_patch(hc)

        cmap = plt.get_cmap('tab10')
        robot_lines = []
        robot_circles = []
        for b in range(B):
            color = cmap(b % 10)
            robot_lines.append(ax.plot([], [], color=color, lw=2, label=f'Robot {b}')[0])
            rc = patches.Circle((999, 999), radius=robot_radius, color=color, alpha=1.0)
            ax.add_patch(rc)
            robot_circles.append(rc)

        # Plot full static paths (faint)
        for h in range(N_humans):
            ax.plot(human_trajs_np[h, :, 0], human_trajs_np[h, :, 1], color=human_color, alpha=0.1, linewidth=1)

        for b, coords in enumerate(robot_trajs_list):
            color = cmap(b % 10)
            ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.15, linewidth=1, linestyle='--')
            if b < len(start_pos_list):
                ax.plot(start_pos_list[b][0], start_pos_list[b][1], marker='s', color=color, markersize=10, alpha=0.8, zorder=5, linestyle='None')
            if b < len(goal_pos_list):
                ax.plot(goal_pos_list[b][0], goal_pos_list[b][1], marker='*', color=color, markersize=14, alpha=0.8, zorder=5, linestyle='None')

        all_pts = [t[:, :2] for t in robot_trajs_list]
        if N_humans > 0:
            all_pts.append(human_trajs_np.reshape(-1, 2))
        all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.array([[-1, -1], [1, 1]])
        margin = 0.5
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        def init():
            for l, c in zip(robot_lines, robot_circles):
                l.set_data([], [])
                c.center = (999, 999)
            for l, c in zip(human_lines, human_circles):
                l.set_data([], [])
                c.center = (999, 999)
            return robot_lines + robot_circles + human_lines + human_circles

        def update(frame):
            for b in range(B):
                traj = robot_trajs_list[b]
                idx = min(frame, traj.shape[0] - 1)
                robot_lines[b].set_data(traj[:idx + 1, 0], traj[:idx + 1, 1])
                robot_circles[b].center = (traj[idx, 0], traj[idx, 1])

            for h in range(N_humans):
                idx = min(frame, H_human - 1)
                human_lines[h].set_data(human_trajs_np[h, :idx + 1, 0], human_trajs_np[h, :idx + 1, 1])
                human_circles[h].center = (human_trajs_np[h, idx, 0], human_trajs_np[h, idx, 1])

            return robot_lines + robot_circles + human_lines + human_circles

        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
        fps = max(1, int(frames / animation_duration))
        ani.save(video_filepath, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f'Saved robot+human gif to {video_filepath}')


def create_animation_video(fig, animate_fn, anim_time=5, n_frames=100, video_filepath='video.gif', **kwargs):
    str_start = "Creating animation"
    # if ".gif" in video_filepath:
    #     video_filepath = video_filepath.replace(".gif", ".gif")
    print(f'{str_start}...')
    ani = FuncAnimation(
        fig,
        animate_fn,
        frames=n_frames,
        interval=anim_time * 1000 / n_frames,
        repeat=False
    )
    print(f'...finished {str_start}')

    str_start = "Saving video..."
    print(f'{str_start}...')
    ani.save(os.path.join(video_filepath), fps=max(1, int(n_frames / anim_time)), dpi=100)
    print(f'...finished {str_start}')


def plot_multiline(ax, X, Y, color='blue', linestyle='solid', **kwargs):
    segments = np.stack((X, Y), axis=-1)
    line_segments = mcoll.LineCollection(segments, colors=[color] * len(segments), linestyle=linestyle)
    ax.add_collection(line_segments)
    points = np.reshape(segments, (-1, 2))
    ax.scatter(points[:, 0], points[:, 1], color=color, s=2 ** 2)
