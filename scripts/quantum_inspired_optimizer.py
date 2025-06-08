"""
Quantum-Inspired Optimizer

Uses quantum computing principles on classical hardware for enhanced optimization.
No actual quantum computer required - uses superposition and entanglement concepts.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import itertools
from concurrent.futures import ProcessPoolExecutor
import random


@dataclass
class QuantumState:
    """Represents a quantum-inspired state."""
    amplitude: complex
    classical_value: Any
    probability: float


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization without quantum hardware."""
    
    def __init__(self, n_qubits: int = 8):
        """Initialize quantum-inspired optimizer.
        
        Args:
            n_qubits: Number of simulated qubits (limits solution space)
        """
        self.n_qubits = n_qubits
        self.max_superposition = 2 ** n_qubits
        
    def quantum_superposition_search(self, 
                                   objective_function: Callable,
                                   search_space: List[Any],
                                   constraints: List[Callable] = None) -> Dict[str, Any]:
        """Search using quantum superposition principle.
        
        Instead of evaluating one solution at a time, evaluate 
        multiple solutions in parallel (superposition).
        
        Args:
            objective_function: Function to optimize
            search_space: List of possible solutions
            constraints: Optional constraint functions
            
        Returns:
            Best solution found
        """
        # Create superposition of states
        superposition_size = min(len(search_space), self.max_superposition)
        
        # Randomly sample states for superposition
        if len(search_space) > superposition_size:
            states = random.sample(search_space, superposition_size)
        else:
            states = search_space
        
        # Create quantum-inspired superposition
        quantum_states = []
        for state in states:
            # Initial amplitude (equal superposition)
            amplitude = complex(1.0 / np.sqrt(len(states)), 0)
            quantum_states.append(QuantumState(
                amplitude=amplitude,
                classical_value=state,
                probability=abs(amplitude) ** 2
            ))
        
        # Evaluate all states in parallel (quantum parallelism)
        with ProcessPoolExecutor() as executor:
            # Evaluate objective function for all states
            futures = []
            for qs in quantum_states:
                if constraints:
                    # Check constraints
                    valid = all(c(qs.classical_value) for c in constraints)
                    if not valid:
                        qs.probability = 0
                        continue
                
                future = executor.submit(objective_function, qs.classical_value)
                futures.append((qs, future))
            
            # Collect results
            results = []
            for qs, future in futures:
                try:
                    score = future.result(timeout=5)
                    results.append((qs, score))
                except Exception as e:
                    print(f"Error evaluating state: {e}")
                    results.append((qs, float('-inf')))
        
        # Quantum measurement (collapse to best state)
        best_state, best_score = max(results, key=lambda x: x[1])
        
        return {
            'best_solution': best_state.classical_value,
            'best_score': best_score,
            'states_evaluated': len(results),
            'superposition_size': superposition_size
        }
    
    def quantum_annealing_optimize(self,
                                 problem_hamiltonian: Callable,
                                 initial_state: Any,
                                 temperature_schedule: List[float] = None) -> Dict[str, Any]:
        """Optimize using quantum annealing principles.
        
        Simulates quantum tunneling to escape local minima.
        
        Args:
            problem_hamiltonian: Energy function to minimize
            initial_state: Starting state
            temperature_schedule: Annealing schedule
            
        Returns:
            Optimized solution
        """
        if temperature_schedule is None:
            temperature_schedule = np.logspace(0, -2, 100)  # 1 to 0.01
        
        current_state = initial_state
        current_energy = problem_hamiltonian(current_state)
        best_state = current_state
        best_energy = current_energy
        
        for temperature in temperature_schedule:
            # Generate neighbor states (quantum fluctuations)
            neighbors = self._generate_quantum_neighbors(current_state)
            
            for neighbor in neighbors:
                neighbor_energy = problem_hamiltonian(neighbor)
                
                # Quantum tunneling probability
                if neighbor_energy < current_energy:
                    # Always accept better solutions
                    current_state = neighbor
                    current_energy = neighbor_energy
                else:
                    # Probabilistic acceptance (quantum tunneling)
                    delta_e = neighbor_energy - current_energy
                    tunneling_prob = np.exp(-delta_e / temperature)
                    
                    if random.random() < tunneling_prob:
                        current_state = neighbor
                        current_energy = neighbor_energy
                
                # Track best solution
                if current_energy < best_energy:
                    best_state = current_state
                    best_energy = current_energy
        
        return {
            'best_solution': best_state,
            'best_energy': best_energy,
            'final_temperature': temperature_schedule[-1]
        }
    
    def quantum_genetic_algorithm(self,
                                fitness_function: Callable,
                                gene_space: List[Any],
                                population_size: int = 50,
                                generations: int = 100) -> Dict[str, Any]:
        """Genetic algorithm with quantum-inspired operators.
        
        Uses quantum superposition for crossover and mutation.
        
        Args:
            fitness_function: Function to maximize
            gene_space: Possible gene values
            population_size: Size of population
            generations: Number of generations
            
        Returns:
            Best individual found
        """
        # Initialize quantum population
        population = []
        for _ in range(population_size):
            # Random chromosome
            chromosome_length = min(self.n_qubits, 16)
            chromosome = [random.choice(gene_space) for _ in range(chromosome_length)]
            
            # Quantum representation
            quantum_chromosome = {
                'classical': chromosome,
                'quantum_state': self._create_quantum_encoding(chromosome),
                'fitness': fitness_function(chromosome)
            }
            population.append(quantum_chromosome)
        
        best_individual = max(population, key=lambda x: x['fitness'])
        
        for generation in range(generations):
            # Quantum-inspired selection
            parents = self._quantum_selection(population)
            
            # Quantum crossover
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._quantum_crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])
            
            # Quantum mutation
            for individual in offspring:
                if random.random() < 0.1:  # 10% mutation rate
                    self._quantum_mutation(individual, gene_space)
            
            # Evaluate fitness
            for individual in offspring:
                individual['fitness'] = fitness_function(individual['classical'])
            
            # Select next generation
            population = self._select_next_generation(population + offspring, population_size)
            
            # Track best
            generation_best = max(population, key=lambda x: x['fitness'])
            if generation_best['fitness'] > best_individual['fitness']:
                best_individual = generation_best
        
        return {
            'best_solution': best_individual['classical'],
            'best_fitness': best_individual['fitness'],
            'generations': generations
        }
    
    def entanglement_optimize(self,
                            objective_function: Callable,
                            variables: List[str],
                            dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Optimize considering entangled variables.
        
        Variables that are entangled (dependent) are optimized together.
        
        Args:
            objective_function: Function to optimize
            variables: List of variable names
            dependencies: Dependencies between variables
            
        Returns:
            Optimal variable assignments
        """
        # Create entanglement graph
        entangled_groups = self._find_entangled_groups(variables, dependencies)
        
        # Optimize each entangled group
        optimal_values = {}
        
        for group in entangled_groups:
            if len(group) == 1:
                # Single variable - classical optimization
                var = group[0]
                best_value = self._optimize_single_variable(var, objective_function)
                optimal_values[var] = best_value
            else:
                # Entangled variables - optimize together
                group_values = self._optimize_entangled_group(
                    group, dependencies, objective_function
                )
                optimal_values.update(group_values)
        
        return {
            'optimal_values': optimal_values,
            'entangled_groups': entangled_groups,
            'objective_value': objective_function(optimal_values)
        }
    
    def _generate_quantum_neighbors(self, state: Any) -> List[Any]:
        """Generate neighbor states using quantum principles."""
        neighbors = []
        
        # Convert state to binary representation
        if isinstance(state, (int, float)):
            # Numeric state
            for delta in [-1, -0.1, 0.1, 1]:
                neighbors.append(state + delta)
        elif isinstance(state, list):
            # List state - modify each element
            for i in range(len(state)):
                for delta in [-1, 1]:
                    neighbor = state.copy()
                    if isinstance(neighbor[i], (int, float)):
                        neighbor[i] += delta
                    neighbors.append(neighbor)
        elif isinstance(state, dict):
            # Dictionary state
            for key in state:
                neighbor = state.copy()
                if isinstance(state[key], (int, float)):
                    neighbor[key] += random.uniform(-1, 1)
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _create_quantum_encoding(self, chromosome: List[Any]) -> np.ndarray:
        """Create quantum encoding of classical chromosome."""
        # Simple encoding: each gene as a qubit angle
        angles = []
        for gene in chromosome:
            if isinstance(gene, (int, float)):
                # Normalize to [0, pi]
                angle = (gene % 360) * np.pi / 180
            else:
                # Hash non-numeric values
                angle = (hash(str(gene)) % 360) * np.pi / 180
            angles.append(angle)
        
        return np.array(angles)
    
    def _quantum_selection(self, population: List[Dict]) -> List[Dict]:
        """Select parents using quantum-inspired probabilities."""
        # Calculate selection probabilities
        fitnesses = np.array([ind['fitness'] for ind in population])
        
        # Quantum-inspired probability distribution
        probabilities = np.abs(fitnesses) / np.sum(np.abs(fitnesses))
        
        # Add quantum noise
        noise = np.random.normal(0, 0.01, len(probabilities))
        probabilities = np.abs(probabilities + noise)
        probabilities /= np.sum(probabilities)
        
        # Select parents
        n_parents = len(population) // 2
        parents_idx = np.random.choice(
            len(population), 
            size=n_parents, 
            p=probabilities,
            replace=False
        )
        
        return [population[i] for i in parents_idx]
    
    def _quantum_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Quantum-inspired crossover operation."""
        # Extract chromosomes
        chrom1 = parent1['classical'].copy()
        chrom2 = parent2['classical'].copy()
        
        # Quantum superposition crossover
        crossover_point = random.randint(1, len(chrom1) - 1)
        
        # Create children with quantum interference
        child1 = chrom1[:crossover_point] + chrom2[crossover_point:]
        child2 = chrom2[:crossover_point] + chrom1[crossover_point:]
        
        # Apply quantum interference at crossover point
        if crossover_point > 0 and crossover_point < len(chrom1):
            # Blend genes at crossover boundary
            if isinstance(child1[crossover_point - 1], (int, float)) and \
               isinstance(child1[crossover_point], (int, float)):
                blend = (child1[crossover_point - 1] + child1[crossover_point]) / 2
                child1[crossover_point - 1] = blend
        
        return (
            {'classical': child1, 'quantum_state': self._create_quantum_encoding(child1)},
            {'classical': child2, 'quantum_state': self._create_quantum_encoding(child2)}
        )
    
    def _quantum_mutation(self, individual: Dict, gene_space: List[Any]) -> None:
        """Apply quantum-inspired mutation."""
        chromosome = individual['classical']
        
        # Select mutation point
        mutation_point = random.randint(0, len(chromosome) - 1)
        
        # Quantum mutation (superposition of current and new value)
        current_gene = chromosome[mutation_point]
        new_gene = random.choice(gene_space)
        
        # Quantum superposition (weighted average for numeric)
        if isinstance(current_gene, (int, float)) and isinstance(new_gene, (int, float)):
            quantum_prob = random.random()
            chromosome[mutation_point] = quantum_prob * current_gene + (1 - quantum_prob) * new_gene
        else:
            chromosome[mutation_point] = new_gene
        
        # Update quantum state
        individual['quantum_state'] = self._create_quantum_encoding(chromosome)
    
    def _select_next_generation(self, all_individuals: List[Dict], size: int) -> List[Dict]:
        """Select next generation using quantum principles."""
        # Sort by fitness
        sorted_individuals = sorted(all_individuals, key=lambda x: x['fitness'], reverse=True)
        
        # Elitism: keep best individuals
        elite_size = size // 4
        next_generation = sorted_individuals[:elite_size]
        
        # Quantum selection for rest
        remaining = sorted_individuals[elite_size:]
        if remaining:
            # Quantum probability based on fitness rank
            ranks = np.arange(len(remaining)) + 1
            probabilities = 1.0 / ranks
            probabilities /= np.sum(probabilities)
            
            selected_idx = np.random.choice(
                len(remaining),
                size=size - elite_size,
                p=probabilities,
                replace=False
            )
            
            next_generation.extend([remaining[i] for i in selected_idx])
        
        return next_generation[:size]
    
    def _find_entangled_groups(self, variables: List[str], 
                              dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Find groups of entangled (dependent) variables."""
        groups = []
        visited = set()
        
        for var in variables:
            if var not in visited:
                group = []
                self._dfs_entanglement(var, dependencies, visited, group)
                groups.append(group)
        
        return groups
    
    def _dfs_entanglement(self, var: str, dependencies: Dict[str, List[str]], 
                         visited: set, group: List[str]) -> None:
        """DFS to find entangled variables."""
        visited.add(var)
        group.append(var)
        
        if var in dependencies:
            for dep in dependencies[var]:
                if dep not in visited:
                    self._dfs_entanglement(dep, dependencies, visited, group)
    
    def _optimize_single_variable(self, var: str, objective_function: Callable) -> Any:
        """Optimize a single variable."""
        # Simple grid search for demo
        best_value = 0
        best_score = float('-inf')
        
        for value in range(-10, 11):
            score = objective_function({var: value})
            if score > best_score:
                best_score = score
                best_value = value
        
        return best_value
    
    def _optimize_entangled_group(self, group: List[str], 
                                 dependencies: Dict[str, List[str]],
                                 objective_function: Callable) -> Dict[str, Any]:
        """Optimize entangled variables together."""
        # Use quantum annealing for entangled optimization
        initial_state = {var: 0 for var in group}
        
        def hamiltonian(state):
            return -objective_function(state)  # Minimize negative for maximization
        
        result = self.quantum_annealing_optimize(hamiltonian, initial_state)
        
        return result['best_solution']


# Example usage
def demonstrate_quantum_optimizer():
    """Demonstrate quantum-inspired optimization."""
    optimizer = QuantumInspiredOptimizer(n_qubits=8)
    
    # Example 1: Task scheduling optimization
    print("=== Quantum Task Scheduling ===")
    
    tasks = [
        {'id': 'A', 'duration': 2, 'priority': 5},
        {'id': 'B', 'duration': 3, 'priority': 3},
        {'id': 'C', 'duration': 1, 'priority': 8},
        {'id': 'D', 'duration': 4, 'priority': 2}
    ]
    
    def scheduling_objective(schedule):
        # Minimize weighted completion time
        total_score = 0
        current_time = 0
        
        for task_id in schedule:
            task = next(t for t in tasks if t['id'] == task_id)
            current_time += task['duration']
            total_score += current_time * task['priority']
        
        return -total_score  # Negative for minimization
    
    # All possible schedules
    from itertools import permutations
    all_schedules = list(permutations(['A', 'B', 'C', 'D']))
    
    result = optimizer.quantum_superposition_search(
        scheduling_objective,
        all_schedules
    )
    
    print(f"Best schedule: {result['best_solution']}")
    print(f"Score: {-result['best_score']}")
    print(f"States evaluated in superposition: {result['states_evaluated']}")
    
    # Example 2: Genetic algorithm for feature selection
    print("\n=== Quantum Genetic Algorithm ===")
    
    def feature_fitness(features):
        # Simulate model accuracy based on features
        return sum(features) - 0.1 * sum(features) ** 2  # Penalize too many features
    
    ga_result = optimizer.quantum_genetic_algorithm(
        feature_fitness,
        gene_space=[0, 1],  # Binary features
        population_size=20,
        generations=50
    )
    
    print(f"Best features: {ga_result['best_solution']}")
    print(f"Fitness: {ga_result['best_fitness']}")
    
    # Example 3: Entangled optimization
    print("\n=== Entangled Variable Optimization ===")
    
    dependencies = {
        'cache_size': ['memory_limit'],
        'memory_limit': ['cache_size', 'thread_count'],
        'thread_count': ['memory_limit']
    }
    
    def system_performance(config):
        # Simulate system performance
        cache = config.get('cache_size', 0)
        memory = config.get('memory_limit', 0)
        threads = config.get('thread_count', 0)
        
        # Performance with constraints
        if memory < cache:
            return -1000  # Invalid
        if threads * 2 > memory:
            return -1000  # Invalid
        
        return cache * 0.5 + memory * 0.3 + threads * 0.2
    
    entangled_result = optimizer.entanglement_optimize(
        system_performance,
        ['cache_size', 'memory_limit', 'thread_count'],
        dependencies
    )
    
    print(f"Optimal configuration: {entangled_result['optimal_values']}")
    print(f"Entangled groups: {entangled_result['entangled_groups']}")
    print(f"Performance: {entangled_result['objective_value']}")


if __name__ == "__main__":
    demonstrate_quantum_optimizer()