import os
import multiprocessing
import json
import traceback
import time
from NoiseEffect.worker import workerFunction
import random
import functools


class NetworkNoiseAnalysis:
    def __init__(self, network_metadata_list, num_instances=1, num_of_random_seeds=50):
        self.network_metadata_list = network_metadata_list
        self.num_instances = num_instances
        self.random_seed_list = random.sample(range(1, 10000), num_of_random_seeds)

        self.results_dict = {}
        self.expanded_requests = (
            self._expandedRequests()
        )  # Create the full list of jobs to run
        self.rescue_file_path = f"rescue_results_{int(time.time())}.json"

    def runAnalysis(self, num_cores=None):
        """
        Runs the analysis for all networks in parallel using a pool of processes.
        """

        if num_cores:
            # If a number is explicitly passed, use it.
            pass
        elif "SLURM_CPUS_PER_TASK" in os.environ:
            # If running on a Slurm cluster, use the number of allocated cores.
            num_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            # If not on Slurm (e.g., local machine), default to using all but one core.
            num_cores = max(1, os.cpu_count() - 1)

        print(f"--- Starting parallel analysis on {num_cores} cores ---")

        try:
            # 'with' statement ensures the pool is properly closed
            with multiprocessing.Pool(processes=num_cores) as pool:
                # pool.map applies the helper function to each item in the list
                # and collects the results in order.
                results_as_list_of_tuples = pool.map(
                    functools.partial(
                        workerFunction, random_seed_list=self.random_seed_list
                    ),
                    self.expanded_requests,
                )

                self.results_dict = dict(results_as_list_of_tuples)

            print("\n--- Parallel analysis complete ---")

        except KeyboardInterrupt:
            print("\n--- Analysis interrupted by user (Ctrl+C) ---")
            self._saveRescueFile("Interrupted by user")
            raise

        except Exception as e:
            print(f"\n--- Analysis failed with error: {e} ---")
            self._saveRescueFile(f"Failed with exception: {e}")
            print(f"--- Traceback: {traceback.format_exc()} ---")
            raise

    def _expandedRequests(self):
        """Creates a new list of requests with specified number of instances."""
        expanded = []
        for request in self.network_metadata_list:
            for i in range(1, self.num_instances + 1):
                # Create a copy and add the instance number
                new_request = request.copy()
                new_request["instance"] = i
                expanded.append(new_request)
        return expanded

    def _saveRescueFile(self, additional_info=""):
        """Save current results to a timestamped rescue file."""
        rescue_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "info": additional_info,
            "completed_analyses": len(self.results_dict),
            "total_planned": len(self.expanded_requests),
            "results": self.results_dict,
        }

        try:
            with open(self.rescue_file_path, "w") as f:
                json.dump(rescue_data, f, indent=2)
            print(f"Rescue file saved: {self.rescue_file_path}")
        except Exception as e:
            print(f"Failed to save rescue file: {e}")
