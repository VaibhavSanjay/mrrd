"""
Modified by Vaibhav Sanjay (2026).
Originally authored by Yorai Shaoul (2024).

MIT License

Copyright (c) 2026 Vaibhav Sanjay
Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.

# Project includes.
from mrrd.common.experiments.experiment_utils import *
from inference_multi_agent import run_multi_agent_trial


def run_multi_agent_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Run the multi-agent planning experiment.
    startt = time.time()
    # Create the experiment config.
    experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Get single trial configs from the experiment config.
    single_trial_configs = experiment_config.get_single_trial_configs_from_experiment_config()
    # So let's run sequentially.
    for single_trial_config in single_trial_configs:
        print(single_trial_config)
        try:
            run_multi_agent_trial(single_trial_config)

            # Aggregate and save data on every step. This is not needed (can be done once at the end).
            combine_and_save_results_for_experiment(experiment_config)
        except Exception as e:
            print("Error in run_multi_agent_experiment: ", e)
            # Save to a file.
            with open(f"error_{experiment_config.time_str}.txt", "a") as f:
                f.write(str(e))
                f.write("This is for single_trial_config: ")
                f.write(str(single_trial_config))
                f.write("\n")
            continue

    # Print the runtime.
    print("Runtime: ", time.time() - startt)
    print("Run: OK.")


if __name__ == "__main__":

    # Instance names. These dictate the maps and start/goals.
    # Create an experiment config.
    experiment_config = MultiAgentPlanningExperimentConfig()
    # Set the experiment config.
    # Number of agents to plan for, run below experiment for each agent count
    experiment_config.num_agents_l = [6]

    # Single tile.
    experiment_config.instance_name = "EnvEmptyNoWait2D-Experiment"

    # Environment to use for planning
    experiment_config.env_id = "EnvEmptyNoWait2D"

    experiment_config.global_model_ids = [[['HumanDiffusion']]]

    # No need to change these
    experiment_config.stagger_start_time_dt = 0
    experiment_config.multi_agent_planner_class_l = ["XECBS"]
    experiment_config.single_agent_planner_class = "MPD"
    experiment_config.runtime_limit = 60 * 10

    # Experiment information
    experiment_config.num_trials_per_combination = 10 # Equal to number of random seeds
    experiment_config.human_trajectory_index = [110, 130, 180, 191, 177] # Example indices with many humans
    experiment_config.random_seeds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # Example random seeds
    experiment_config.render_animation = True

    # Run the experiment.
    run_multi_agent_experiment(experiment_config)
