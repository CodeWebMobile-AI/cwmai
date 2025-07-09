from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import time
import os
import random
import socket
import psutil  # Requires installation: pip install psutil
import subprocess  # For executing shell commands safely
from scripts.logger import get_logger
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfigManager  # Assuming this exists


class RestartSystem:
    """
    A tool to simulate a system restart with error handling, logging, caching,
    and additional utility functions.
    """

    _restart_cache = {}  # Class-level cache
    _system_metrics_cache = {} # Cache for system metrics
    _process_cache = {}  # Cache for process listing
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 5
    DEFAULT_CACHE_EXPIRATION = 60
    DEFAULT_SYSTEM_METRICS_CACHE_EXPIRATION = 30
    DEFAULT_PROCESS_CACHE_EXPIRATION = 60


    def __init__(self, config_path: str = "config.yaml", retry_attempts: int = DEFAULT_RETRY_ATTEMPTS, retry_delay: int = DEFAULT_RETRY_DELAY,
                 cache_expiration: int = DEFAULT_CACHE_EXPIRATION, system_metrics_cache_expiration: int = DEFAULT_SYSTEM_METRICS_CACHE_EXPIRATION,
                 process_cache_expiration: int = DEFAULT_PROCESS_CACHE_EXPIRATION):
        """
        Initializes the RestartSystem tool.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to "config.yaml".
            retry_attempts (int, optional): Number of retry attempts for restart. Defaults to 3.
            retry_delay (int, optional): Delay in seconds between retry attempts. Defaults to 5.
            cache_expiration (int, optional): Time in seconds after which cached results expire. Defaults to 60.
            system_metrics_cache_expiration (int, optional):  Time in seconds after which system metrics cache expires. Defaults to 30.
            process_cache_expiration (int, optional): Time in seconds after which process listing cache expires. Defaults to 60.
        """
        self.logger = get_logger(__name__)
        try:
            self.config_manager = MCPConfigManager(config_path)  # Assuming this can handle config loading errors.
            self.config = self.config_manager.get_config() #Access the loaded config
        except Exception as e:
            self.logger.critical(f"Failed to load configuration: {e}")
            raise  # Re-raise the exception to prevent the tool from running with a broken config.
        self.error_handler = ErrorHandler(self.logger)
        self.retry_attempts = self._validate_retry_attempts(retry_attempts)
        self.retry_delay = self._validate_retry_delay(retry_delay)
        self.cache_expiration = self._validate_cache_expiration(cache_expiration)
        self.system_metrics_cache_expiration = self._validate_system_metrics_cache_expiration(system_metrics_cache_expiration)
        self.process_cache_expiration = self._validate_process_cache_expiration(process_cache_expiration)


    def _validate_retry_attempts(self, attempts: int) -> int:
        """Validates the retry attempts value."""
        if not isinstance(attempts, int) or attempts <= 0:
            self.logger.warning(f"Invalid retry_attempts value: {attempts}. Using default value of {self.DEFAULT_RETRY_ATTEMPTS}.")
            return self.DEFAULT_RETRY_ATTEMPTS
        return attempts

    def _validate_retry_delay(self, delay: int) -> int:
        """Validates the retry delay value."""
        if not isinstance(delay, int) or delay < 0:
            self.logger.warning(f"Invalid retry_delay value: {delay}. Using default value of {self.DEFAULT_RETRY_DELAY}.")
            return self.DEFAULT_RETRY_DELAY
        return delay

    def _validate_cache_expiration(self, expiration: int) -> int:
        """Validates the cache expiration value."""
        if not isinstance(expiration, int) or expiration < 0:
            self.logger.warning(f"Invalid cache_expiration value: {expiration}. Using default value of {self.DEFAULT_CACHE_EXPIRATION}.")
            return self.DEFAULT_CACHE_EXPIRATION
        return expiration

    def _validate_system_metrics_cache_expiration(self, expiration: int) -> int:
        """Validates the system metrics cache expiration value."""
        if not isinstance(expiration, int) or expiration < 0:
            self.logger.warning(f"Invalid system_metrics_cache_expiration value: {expiration}. Using default value of {self.DEFAULT_SYSTEM_METRICS_CACHE_EXPIRATION}.")
            return self.DEFAULT_SYSTEM_METRICS_CACHE_EXPIRATION
        return expiration

    def _validate_process_cache_expiration(self, expiration: int) -> int:
        """Validates the process cache expiration value."""
        if not isinstance(expiration, int) or expiration < 0:
            self.logger.warning(f"Invalid process_cache_expiration value: {expiration}. Using default value of {self.DEFAULT_PROCESS_CACHE_EXPIRATION}.")
            return self.DEFAULT_PROCESS_CACHE_EXPIRATION
        return expiration

    def clear_cache(self, cache_type: str = "all"):
        """Clears the specified cache(s)."""
        if cache_type in ("all", "restart"):
            RestartSystem._restart_cache = {}
            print("Restart cache cleared.")
        if cache_type in ("all", "system_metrics"):
            RestartSystem._system_metrics_cache = {}
            print("System Metrics cache cleared.")
        if cache_type in ("all", "process"):
            RestartSystem._process_cache = {}
            print("Process cache cleared.")
        if cache_type == "all":
            print("All caches cleared.")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Simulates getting the system status.  This is a placeholder for actual
        system status retrieval.  Uses the configuration to determine probability of offline status.

        Returns:
            Dict[str, Any]: A dictionary containing the system status.
                             'status' will be 'online' or 'offline'.
                             'timestamp' will contain the time the status was checked.
        """
        try:
            offline_probability = self.config.get('system_offline_probability', 0.1)  # Default 10% chance
            if not 0 <= offline_probability <= 1:
                self.logger.warning(f"Invalid system_offline_probability in config: {offline_probability}. Using default 0.1.")
                offline_probability = 0.1

            if random.random() < offline_probability:
                status = 'offline'
            else:
                status = 'online'

            timestamp = time.time()
            return {'status': status, 'timestamp': timestamp}
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            self.error_handler.handle_error(f"Error getting system status: {e}", error_category="System Status", error_severity="Medium")
            return {'status': 'unknown', 'timestamp': time.time(), 'error': str(e)}


    def _is_cache_valid(self, cache_key: str, cache_type: str = "restart") -> bool:
        """Checks if the cached result is still valid based on cache_expiration."""
        cache = RestartSystem._restart_cache if cache_type == "restart" else \
                RestartSystem._system_metrics_cache if cache_type == "system_metrics" else RestartSystem._process_cache

        expiration = self.cache_expiration if cache_type == "restart" else \
                     self.system_metrics_cache_expiration if cache_type == "system_metrics" else self.process_cache_expiration

        if cache_key not in cache:
            return False

        cached_data = cache[cache_key]
        if 'timestamp' not in cached_data:
            return False  # Invalidate if timestamp is missing

        if (time.time() - cached_data['timestamp']) > expiration:
            self.logger.debug(f"Cache for key '{cache_key}' of type '{cache_type}' expired.")
            return False

        return True

    def restart(self, force: bool = False, cache_key: str = "default") -> Dict[str, Any]:
        """
        Simulates restarting the system with retry mechanism, caching, and force option.

        Args:
            force (bool, optional): Forces a restart even if the system appears healthy. Defaults to False.
            cache_key (str, optional): Key to use for caching the restart result. Defaults to "default".

        Returns:
            Dict[str, Any]: A dictionary containing the result of the operation.
                            'success' will be True if the restart was successful, False otherwise.
                            'message' will contain a success or informational message.
                            'error' will contain an error message if the restart failed.
                            'timestamp' will contain the time the restart was attempted.
                            'hostname' will contain the hostname of the machine.
                            'pid' will contain the process ID of the script.
                            'attempt' will contain the number of attempts made.
        """

        if cache_key in RestartSystem._restart_cache and not force and self._is_cache_valid(cache_key, cache_type="restart"):
            print(f"Returning cached restart result for key: {cache_key}")
            cached_result = RestartSystem._restart_cache[cache_key]
            cached_result['cached'] = True  # Indicate that this is a cached result.
            return cached_result

        timestamp = time.time()  # Record the time of the restart attempt.
        result = {} # Initialize result dictionary

        try:
            print(f"Attempting to restart the system (force={force}).")

            if not force:
                system_status = self.get_system_status()
                if system_status['status'] == 'online':
                    print("System appears healthy, skipping restart (unless force=True).")
                    result = {"success": True, "message": "System already online, restart skipped.",
                              'timestamp': timestamp, 'cached': False, 'hostname': self.get_hostname(), 'pid': self.get_pid()}
                    RestartSystem._restart_cache[cache_key] = result
                    return result

            for attempt in range(self.retry_attempts):
                print(f"Restart attempt {attempt + 1}/{self.retry_attempts}")
                try:
                    # Simulate a restart process (replace with actual restart logic)
                    print("System restart initiated (simulated).")
                    restart_command = self.config.get("restart_command", "echo 'Simulated restart'") #Get command from config
                    #Safely execute shell command
                    process = subprocess.Popen(restart_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
                    stdout, stderr = process.communicate()
                    return_code = process.returncode

                    if return_code == 0:
                        print(f"System restart completed successfully (simulated): {stdout.decode()}")
                        result = {"success": True, "message": f"System restarted successfully. Output: {stdout.decode()}",
                                  'timestamp': timestamp, 'cached': False, 'attempt': attempt + 1, 'hostname': self.get_hostname(), 'pid': self.get_pid()} # Include attempt number
                        RestartSystem._restart_cache[cache_key] = result
                        return result
                    else:
                        error_message = f"System restart failed (simulated). Return code: {return_code}, Error: {stderr.decode()}"
                        self.logger.error(error_message)
                        result = {"success": False, "error": error_message, 'timestamp': timestamp, 'cached': False, 'attempt': attempt + 1, 'hostname': self.get_hostname(), 'pid': self.get_pid()} #Include attempt number

                except FileNotFoundError as e: #Catch if the shell does not exist
                    error_message = f"Shell not found during restart attempt {attempt + 1}: {e}.  Check your system configuration."
                    self.logger.error(error_message)
                    self.error_handler.handle_error(error_message, error_category="System Restart", error_severity="Critical")
                    result = {"success": False, "error": error_message, 'timestamp': timestamp, 'cached': False, 'attempt': attempt + 1, 'hostname': self.get_hostname(), 'pid': self.get_pid()}

                except Exception as e:
                    error_message = f"Error during restart attempt {attempt + 1}: {e}"
                    self.logger.error(error_message)
                    self.error_handler.handle_error(error_message, error_category="System Restart",
                                                       error_severity="Medium")  # Less severe than a complete failure
                    result = {"success": False, "error": error_message, 'timestamp': timestamp, 'cached': False, 'attempt': attempt + 1, 'hostname': self.get_hostname(), 'pid': self.get_pid()} # Include attempt number

                if attempt < self.retry_attempts - 1:
                    print(f"Waiting {self.retry_delay} seconds before next retry...")
                    time.sleep(self.retry_delay)

            self.logger.error("System restart failed after multiple retries.")
            self.error_handler.handle_error("System restart failed after retries", error_category="System Restart",
                                               error_severity="Critical")
            result = {"success": False, "error": "System restart failed after multiple retries.",
                      'timestamp': timestamp, 'cached': False, 'attempt': self.retry_attempts, 'hostname': self.get_hostname(), 'pid': self.get_pid()} # Include attempt number
            RestartSystem._restart_cache[cache_key] = result  # Cache the failure.
            return result

        except Exception as e:
            error_message = f"An unexpected error occurred during system restart: {e}"
            self.logger.exception(error_message)
            self.error_handler.handle_error(error_message, error_category="System Restart",
                                               error_severity="Critical")
            result = {"success": False, "error": error_message, 'timestamp': timestamp, 'cached': False, 'hostname': self.get_hostname(), 'pid': self.get_pid()}
            return result
        finally:
            if 'success' not in result:  # Ensure result is always populated, even on unexpected exceptions.
                result = {"success": False, "error": "Unknown error during restart.", 'timestamp': timestamp, 'cached': False, 'hostname': self.get_hostname(), 'pid': self.get_pid()}
            result['hostname'] = self.get_hostname()  # Add hostname to output for context
            result['pid'] = self.get_pid() # Add pid to output for context

    def get_pid(self) -> Optional[int]:
        """
        Gets the current process ID.

        Returns:
            Optional[int]: The process ID, or None if it cannot be retrieved.
        """
        try:
            pid = os.getpid()
            self.logger.debug(f"Process ID: {pid}")  # Log the PID at debug level.
            return pid
        except OSError as e:
            self.logger.error(f"Failed to retrieve process ID: {e}")
            self.error_handler.handle_error(f"Failed to retrieve process ID: {e}", error_category="System Info",
                                               error_severity="Low")  # Not critical
            return None

    def get_hostname(self) -> Optional[str]:
        """
        Gets the hostname of the system.

        Returns:
            Optional[str]: The hostname, or None if it cannot be retrieved.
        """
        try:
            hostname = socket.gethostname()
            self.logger.debug(f"Hostname: {hostname}")
            return hostname
        except OSError as e:
            self.logger.error(f"Failed to retrieve hostname: {e}")
            self.error_handler.handle_error(f"Failed to retrieve hostname: {e}", error_category="System Info",
                                               error_severity="Low")
            return None

    def is_port_open(self, port: int, host: str = 'localhost') -> bool:
        """
        Checks if a given port is open on the system.

        Args:
            port (int): The port number to check.
            host (str, optional): The host to connect to. Defaults to 'localhost'.

        Returns:
            bool: True if the port is open, False otherwise.
        """
        if not isinstance(port, int):
            self.logger.error(f"Invalid port type: {type(port)}. Port must be an integer.")
            self.error_handler.handle_error(f"Invalid port type: {type(port)}", error_category="Network", error_severity="Medium")
            return False

        if not 1 <= port <= 65535:
            self.logger.error(f"Invalid port number: {port}. Port must be between 1 and 65535.")
            self.error_handler.handle_error(f"Invalid port number: {port}", error_category="Network", error_severity="Medium")
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)  # Short timeout
                s.connect((host, port))
                self.logger.debug(f"Port {port} on {host} is open.")
                return True
        except (socket.error, socket.timeout) as e:
            self.logger.debug(f"Port {port} on {host} is closed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error checking port {port} on {host}: {e}")
            self.error_handler.handle_error(f"Error checking port {port} on {host}: {e}", error_category="Network",
                                               error_severity="Medium")  # Could indicate a service issue
            return False

    def get_system_metrics(self, cache_key: str = "system_metrics") -> Dict[str, Any]:
        """
        Retrieves system metrics like CPU usage, memory usage, and disk usage.
        Uses caching to avoid frequent calls to psutil.

        Args:
            cache_key (str, optional): Key to use for caching the system metrics. Defaults to "system_metrics".

        Returns:
            Dict[str, Any]: A dictionary containing the system metrics.
                            Returns an empty dictionary if an error occurs.
        """

        if cache_key in RestartSystem._system_metrics_cache and self._is_cache_valid(cache_key, cache_type="system_metrics"):
            print(f"Returning cached system metrics for key: {cache_key}")
            cached_metrics = RestartSystem._system_metrics_cache[cache_key]
            cached_metrics['cached'] = True
            return cached_metrics

        try:
            cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU usage over 1 second
            memory_usage = psutil.virtual_memory().percent  # Get memory usage percentage
            disk_usage = psutil.disk_usage('/').percent  # Get disk usage percentage for root partition
            boot_time = psutil.boot_time()  #Get system boot time
            timestamp = time.time()

            metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'boot_time': boot_time,
                'timestamp': timestamp,
                'cached': False
            }

            RestartSystem._system_metrics_cache[cache_key] = metrics
            return metrics

        except Exception as e:
            error_message = f"Error retrieving system metrics: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, error_category="System Info", error_severity="High")
            return {}  # Return an empty dictionary to indicate failure.

    def get_running_processes(self, process_name: Optional[str] = None, cache_key: str = "processes") -> List[Dict[str, Any]]:
        """
        Retrieves a list of running processes.  Optionally filters by process name. Caches the full process list.

        Args:
            process_name (str, optional):  If provided, only processes with this name are returned.  Defaults to None.
            cache_key (str, optional): Key to use for caching the process list. Defaults to "processes".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a process.
                                  Each dictionary contains 'pid', 'name', and 'cpu_percent'.
                                  Returns an empty list if no processes are found or an error occurs.
        """

        if cache_key in RestartSystem._process_cache and self._is_cache_valid(cache_key, cache_type="process"):
            print(f"Returning cached process list for key: {cache_key}")
            cached_processes = RestartSystem._process_cache[cache_key]['processes'] # Extract list from cache
            cached_processes_filtered = [p for p in cached_processes if process_name is None or p['name'] == process_name] # Filter in memory.
            return cached_processes_filtered


        processes_list = []
        try:
            for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'username', 'memory_info']): #added more info
                try: #Handle individual process access errors
                    process_info = process.info
                    processes_list.append(process_info) #No filtering happens here anymore, caching happens first.
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.warning(f"Could not retrieve info for process {process.pid}: {e}") #Log but continue

            #Cache the full list, then filter.
            RestartSystem._process_cache[cache_key] = {'processes': processes_list, 'timestamp': time.time()}

            filtered_processes = [p for p in processes_list if process_name is None or p['name'] == process_name]


            if not filtered_processes:
                print(f"No running processes found matching name: {process_name}")
            else:
                self.logger.debug(f"Found {len(filtered_processes)} running processes matching name: {process_name}")

            return filtered_processes

        except Exception as e:
            error_message = f"Error retrieving running processes: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, error_category="System Info", error_severity="Medium")
            return []

    def get_file_content(self, file_path: str, max_lines: int = 100) -> Optional[List[str]]:
        """
        Reads a file and returns its content as a list of lines.

        Args:
            file_path (str): The path to the file.
            max_lines (int, optional): The maximum number of lines to read. Defaults to 100.

        Returns:
            Optional[List[str]]: A list of strings, where each string is a line from the file.
                                Returns None if the file does not exist or an error occurs.
        """
        try:
            file_path_obj = Path(file_path).resolve()  # Resolve the path to its absolute form

            if not file_path_obj.exists():
                self.logger.error(f"File not found: {file_path}")
                self.error_handler.handle_error(f"File not found: {file_path}", error_category="File Access", error_severity="Medium")
                return None

            if not file_path_obj.is_file():
                self.logger.error(f"Not a valid file: {file_path}")
                self.error_handler.handle_error(f"Not a valid file: {file_path}", error_category="File Access", error_severity="Medium")
                return None


            with open(file_path_obj, 'r') as f:
                lines = [next(f).strip() for _ in range(max_lines)] #Efficiently read up to max_lines
                return lines

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            self.error_handler.handle_error(f"Error reading file {file_path}: {e}", error_category="File Access", error_severity="High")
            return None


if __name__ == "__main__":
    # Example usage:
    restart_tool = RestartSystem(retry_attempts=2, retry_delay=3, cache_expiration=10, system_metrics_cache_expiration=5, process_cache_expiration=15) #Reduced cache expiration for demonstration.

    # Demonstrates clearing cache
    restart_tool.clear_cache()

    # Demonstrates getting the pid
    pid = restart_tool.get_pid()
    if pid:
        print(f"Current process ID: {pid}")
    else:
        print("Could not retrieve process ID.")

    # Demonstrates getting the hostname
    hostname = restart_tool.get_hostname()
    if hostname:
        print(f"Current hostname: {hostname}")
    else:
        print("Could not retrieve hostname.")

    # Demonstrates port check
    port_to_check = 80  # Example port
    if restart_tool.is_port_open(port_to_check):
        print(f"Port {port_to_check} is open.")
    else:
        print(f"Port {port_to_check} is closed.")

    # Demonstrates port check with different host
    port_to_check = 8080  # Example port
    if restart_tool.is_port_open(port_to_check, host="google.com"):
        print(f"Port {port_to_check} on google.com is open.")
    else:
        print(f"Port {port_to_check} on google.com is closed.")


    # Get and print system metrics
    system_metrics = restart_tool.get_system_metrics()
    if system_metrics:
        print("System Metrics:")
        print(f"  CPU Usage: {system_metrics['cpu_usage']}%")
        print(f"  Memory Usage: {system_metrics['memory_usage']}%")
        print(f"  Disk Usage: {system_metrics['disk_usage']}%")
        print(f"  Boot Time: {time.ctime(system_metrics['boot_time'])}")
    else:
        print("Could not retrieve system metrics.")

    # First attempt
    result = restart_tool.restart()

    if result.get("success"):
        print(f"System restart was successful: {result.get('message')}")
    else:
        print(f"System restart failed: {result.get('error')}")

    # Second attempt (should return cached result if not forced)
    result2 = restart_tool.restart()

    if result2.get("success"):
        print(f"System restart was successful (cached): {result2.get('message')}")
        if result2.get("cached"):
            print("Result was retrieved from cache.")
    else:
        print(f"System restart failed (cached): {result2.get('error')}")

    # Force restart
    result3 = restart_tool.restart(force=True)

    if result3.get("success"):
        print(f"System restart was successful (forced): {result3.get('message')}")
    else:
        print(f"System restart failed (forced): {result3.get('error')}")

    # Wait for cache to expire and try again
    print("Waiting for cache to expire...")
    time.sleep(11)

    result4 = restart_tool.restart()
    if result4.get("success"):
        print(f"System restart was successful: {result4.get('message')}")
        if result4.get("cached"):
            print("Result was retrieved from cache.")
        else:
            print("Result was NOT retrieved from cache.") #This will be printed because cache has expired
    else:
        print(f"System restart failed: {result4.get('error')}")

    # Demonstrate getting system metrics (cached)
    system_metrics2 = restart_tool.get_system_metrics()
    if system_metrics2:
        print("System Metrics (Cached):")
        print(f"  CPU Usage: {system_metrics2['cpu_usage']}%")
        print(f"  Memory Usage: {system_metrics2['memory_usage']}%")
        print(f"  Disk Usage: {system_metrics2['disk_usage']}%")
        print(f"  Boot Time: {time.ctime(system_metrics2['boot_time'])}")
        if system_metrics2.get("cached"):
            print("System metrics were retrieved from cache.")
        else:
             print("System metrics were NOT retrieved from cache.")


    # Demonstrate getting system metrics after cache expiration
    print("Waiting for system metrics cache to expire...")
    time.sleep(6)  # Wait longer than system_metrics_cache_expiration

    system_metrics3 = restart_tool.get_system_metrics()
    if system_metrics3:
        print("System Metrics (After Cache Expiration):")
        print(f"  CPU Usage: {system_metrics3['cpu_usage']}%")
        print(f"  Memory Usage: {system_metrics3['memory_usage']}%")
        print(f"  Disk Usage: {system_metrics3['disk_usage']}%")
        print(f"  Boot Time: {time.ctime(system_metrics3['boot_time'])}")
        if system_metrics3.get("cached"):
            print("System metrics were retrieved from cache.")
        else:
            print("System metrics were NOT retrieved from cache.")

    # Demonstrate getting running processes
    running_processes = restart_tool.get_running_processes()
    if running_processes:
        print("Running Processes:")
        for process in running_processes:
            print(f"  PID: {process['pid']}, Name: {process['name']}, CPU%: {process['cpu_percent']}%, User: {process.get('username', 'N/A')}, Memory: {process['memory_info'].rss / (1024 * 1024):.2f} MB") #Added user and memory
    else:
        print("Could not retrieve running processes.")

    # Demonstrate filtering processes by name
    python_processes = restart_tool.get_running_processes(process_name="python")
    if python_processes:
        print("Python Processes:")
        for process in python_processes:
            print(f"  PID: {process['pid']}, Name: {process['name']}, CPU%: {process['cpu_percent']}%, User: {process.get('username', 'N/A')}, Memory: {process['memory_info'].rss / (1024 * 1024):.2f} MB") #Added user and memory

    else:
        print("No Python processes found.")

    #Demonstrates caching running processes and filtering from cache.
    print("Waiting for process list cache to expire...")
    time.sleep(16)

    python_processes2 = restart_tool.get_running_processes(process_name="python")
    if python_processes2:
        print("Python Processes (Cached?):")
        for process in python_processes2:
            print(f"  PID: {process['pid']}, Name: {process['name']}, CPU%: {process['cpu_percent']}%, User: {process.get('username', 'N/A')}, Memory: {process['memory_info'].rss / (1024 * 1024):.2f} MB") #Added user and memory

    else:
        print("No Python processes found.")


    # Demonstrate getting file content
    log_file = "example.log" #Replace with an actual log file for testing.
    #Create a dummy file
    with open(log_file, "w") as f:
        for i in range(200):
            f.write(f"Line {i+1}: This is a test log line.\n")
    file_content = restart_tool.get_file_content(log_file, max_lines=50)

    if file_content:
        print(f"Content of {log_file} (first 50 lines):")
        for line in file_content:
            print(line)
    else:
        print(f"Could not retrieve content of {log_file}.")
