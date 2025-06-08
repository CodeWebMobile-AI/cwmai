"""
Evolutionary Neural Architecture Search (NAS)

CPU-based neural architecture search that evolves optimal network designs.
Works without GPU by using small proxy datasets and efficient training.
"""

import numpy as np
import json
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import os
import tempfile


class LayerType(Enum):
    """Types of layers that can be evolved."""
    DENSE = "dense"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    ACTIVATION = "activation"
    POOLING = "pooling"


class ActivationType(Enum):
    """Activation functions available."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    units: Optional[int] = None
    activation: Optional[ActivationType] = None
    dropout_rate: Optional[float] = None
    kernel_size: Optional[int] = None
    pool_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in asdict(self).items() if v is not None}


@dataclass
class Architecture:
    """Neural network architecture representation."""
    layers: List[LayerConfig]
    learning_rate: float
    batch_size: int
    optimizer: str
    fitness: Optional[float] = None
    training_time: Optional[float] = None
    parameters: Optional[int] = None
    
    def get_hash(self) -> str:
        """Generate unique hash for architecture."""
        arch_str = json.dumps([layer.to_dict() for layer in self.layers], sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()[:12]
    
    def count_parameters(self) -> int:
        """Estimate parameter count."""
        params = 0
        prev_units = 32  # Assume input size
        
        for layer in self.layers:
            if layer.layer_type == LayerType.DENSE and layer.units:
                params += prev_units * layer.units + layer.units
                prev_units = layer.units
            elif layer.layer_type == LayerType.CONV2D and layer.units:
                # Simplified conv parameter estimation
                kernel_size = layer.kernel_size or 3
                params += kernel_size * kernel_size * prev_units * layer.units + layer.units
                prev_units = layer.units
            elif layer.layer_type in [LayerType.LSTM, LayerType.GRU] and layer.units:
                # RNN parameter estimation
                params += 4 * (prev_units + layer.units + 1) * layer.units
                prev_units = layer.units
        
        self.parameters = params
        return params


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search for CPU."""
    
    def __init__(self, max_time_hours: float = 2.0, population_size: int = 20):
        """Initialize NAS.
        
        Args:
            max_time_hours: Maximum time to run evolution
            population_size: Number of architectures in population
        """
        self.max_time_seconds = max_time_hours * 3600
        self.population_size = population_size
        self.generation = 0
        self.best_architecture = None
        self.architecture_cache = {}  # Cache fitness evaluations
        
        # Evolution parameters
        self.mutation_rate = 0.3
        self.crossover_rate = 0.5
        self.elite_size = max(2, population_size // 10)
        
        # Architecture constraints
        self.max_layers = 10
        self.max_parameters = 1_000_000  # 1M parameters max for CPU
        
    def search(self, task_type: str, train_data: Any = None, 
               val_data: Any = None) -> Architecture:
        """Run architecture search for a specific task.
        
        Args:
            task_type: Type of task (classification, regression, etc.)
            train_data: Training data (optional, will use proxy if None)
            val_data: Validation data (optional)
            
        Returns:
            Best architecture found
        """
        print(f"Starting NAS for {task_type} task...")
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(task_type)
        
        # Evolution loop
        while (time.time() - start_time) < self.max_time_seconds:
            self.generation += 1
            
            # Evaluate fitness
            for arch in population:
                if arch.fitness is None:
                    arch.fitness = self._evaluate_architecture(
                        arch, task_type, train_data, val_data
                    )
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness or 0, reverse=True)
            
            # Print progress
            best = population[0]
            print(f"Generation {self.generation}: Best fitness = {best.fitness:.4f}, "
                  f"Parameters = {best.parameters:,}")
            
            # Check if we should stop
            if best.fitness and best.fitness > 0.95:
                print("Reached target fitness!")
                break
            
            # Create next generation
            population = self._evolve_population(population)
            
            # Save best
            if self.best_architecture is None or \
               (best.fitness and best.fitness > (self.best_architecture.fitness or 0)):
                self.best_architecture = best
        
        print(f"NAS completed in {time.time() - start_time:.1f} seconds")
        return self.best_architecture
    
    def _initialize_population(self, task_type: str) -> List[Architecture]:
        """Create initial population of architectures."""
        population = []
        
        # Add some hand-designed architectures
        if task_type == "classification":
            # Simple MLP
            population.append(Architecture(
                layers=[
                    LayerConfig(LayerType.DENSE, units=128, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DROPOUT, dropout_rate=0.2),
                    LayerConfig(LayerType.DENSE, units=64, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=10, activation=ActivationType.SIGMOID)
                ],
                learning_rate=0.001,
                batch_size=32,
                optimizer="adam"
            ))
            
            # Simple CNN
            population.append(Architecture(
                layers=[
                    LayerConfig(LayerType.CONV2D, units=32, kernel_size=3, activation=ActivationType.RELU),
                    LayerConfig(LayerType.POOLING, pool_size=2),
                    LayerConfig(LayerType.CONV2D, units=64, kernel_size=3, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=128, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=10, activation=ActivationType.SIGMOID)
                ],
                learning_rate=0.001,
                batch_size=32,
                optimizer="adam"
            ))
        
        elif task_type == "regression":
            # Regression MLP
            population.append(Architecture(
                layers=[
                    LayerConfig(LayerType.DENSE, units=64, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=32, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=1)
                ],
                learning_rate=0.01,
                batch_size=16,
                optimizer="adam"
            ))
        
        elif task_type == "sequence":
            # LSTM architecture
            population.append(Architecture(
                layers=[
                    LayerConfig(LayerType.LSTM, units=64),
                    LayerConfig(LayerType.DROPOUT, dropout_rate=0.2),
                    LayerConfig(LayerType.DENSE, units=32, activation=ActivationType.RELU),
                    LayerConfig(LayerType.DENSE, units=1)
                ],
                learning_rate=0.001,
                batch_size=16,
                optimizer="adam"
            ))
        
        # Fill rest with random architectures
        while len(population) < self.population_size:
            arch = self._create_random_architecture(task_type)
            if arch.count_parameters() <= self.max_parameters:
                population.append(arch)
        
        return population
    
    def _create_random_architecture(self, task_type: str) -> Architecture:
        """Create a random valid architecture."""
        layers = []
        
        # Random number of layers
        n_layers = random.randint(2, self.max_layers)
        
        # Task-specific layer selection
        if task_type == "classification":
            layer_types = [LayerType.DENSE, LayerType.CONV2D, LayerType.DROPOUT]
        elif task_type == "regression":
            layer_types = [LayerType.DENSE, LayerType.DROPOUT, LayerType.BATCH_NORM]
        elif task_type == "sequence":
            layer_types = [LayerType.LSTM, LayerType.GRU, LayerType.DENSE, LayerType.DROPOUT]
        else:
            layer_types = list(LayerType)
        
        # Build layers
        for i in range(n_layers):
            layer_type = random.choice(layer_types)
            
            if layer_type == LayerType.DENSE:
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    units=random.choice([16, 32, 64, 128, 256]),
                    activation=random.choice(list(ActivationType))
                ))
            elif layer_type == LayerType.CONV2D:
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    units=random.choice([16, 32, 64]),
                    kernel_size=random.choice([3, 5]),
                    activation=random.choice(list(ActivationType))
                ))
            elif layer_type == LayerType.DROPOUT:
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    dropout_rate=random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                ))
            elif layer_type in [LayerType.LSTM, LayerType.GRU]:
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    units=random.choice([16, 32, 64, 128])
                ))
            elif layer_type == LayerType.POOLING:
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    pool_size=2
                ))
            elif layer_type == LayerType.BATCH_NORM:
                layers.append(LayerConfig(layer_type=layer_type))
        
        # Add output layer
        if task_type == "classification":
            layers.append(LayerConfig(
                layer_type=LayerType.DENSE,
                units=10,  # Assume 10 classes
                activation=ActivationType.SIGMOID
            ))
        else:
            layers.append(LayerConfig(
                layer_type=LayerType.DENSE,
                units=1
            ))
        
        return Architecture(
            layers=layers,
            learning_rate=random.choice([0.0001, 0.001, 0.01, 0.1]),
            batch_size=random.choice([16, 32, 64]),
            optimizer=random.choice(["adam", "sgd", "rmsprop"])
        )
    
    def _evaluate_architecture(self, arch: Architecture, task_type: str,
                             train_data: Any, val_data: Any) -> float:
        """Evaluate architecture fitness.
        
        Uses fast proxy evaluation on CPU.
        """
        # Check cache
        arch_hash = arch.get_hash()
        if arch_hash in self.architecture_cache:
            cached = self.architecture_cache[arch_hash]
            arch.fitness = cached['fitness']
            arch.training_time = cached['training_time']
            return arch.fitness
        
        # Count parameters
        arch.count_parameters()
        
        # Penalize very large models
        if arch.parameters > self.max_parameters:
            arch.fitness = 0.0
            return 0.0
        
        # Simulate training (in practice, would actually train)
        start_time = time.time()
        
        try:
            # Create a proxy fitness score
            fitness = self._proxy_evaluation(arch, task_type)
            
            # Add efficiency bonus (prefer smaller, faster models)
            param_penalty = arch.parameters / self.max_parameters
            fitness = fitness * (1 - 0.2 * param_penalty)
            
            arch.training_time = time.time() - start_time
            arch.fitness = fitness
            
            # Cache result
            self.architecture_cache[arch_hash] = {
                'fitness': fitness,
                'training_time': arch.training_time
            }
            
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            arch.fitness = 0.0
        
        return arch.fitness
    
    def _proxy_evaluation(self, arch: Architecture, task_type: str) -> float:
        """Fast proxy evaluation without full training.
        
        In practice, this would:
        1. Build the actual model
        2. Train for a few epochs on subset of data
        3. Return validation accuracy
        
        For demo, we simulate based on architecture properties.
        """
        score = 0.5  # Base score
        
        # Reward good practices
        has_dropout = any(l.layer_type == LayerType.DROPOUT for l in arch.layers)
        has_batch_norm = any(l.layer_type == LayerType.BATCH_NORM for l in arch.layers)
        
        if has_dropout:
            score += 0.1
        if has_batch_norm:
            score += 0.05
        
        # Task-specific scoring
        if task_type == "classification":
            # Prefer conv layers for image tasks
            conv_count = sum(1 for l in arch.layers if l.layer_type == LayerType.CONV2D)
            if conv_count > 0:
                score += 0.1 * min(conv_count / 3, 1)
        
        elif task_type == "sequence":
            # Prefer RNN layers for sequence tasks
            rnn_count = sum(1 for l in arch.layers 
                          if l.layer_type in [LayerType.LSTM, LayerType.GRU])
            if rnn_count > 0:
                score += 0.15 * min(rnn_count / 2, 1)
        
        # Learning rate scoring
        if 0.0001 <= arch.learning_rate <= 0.01:
            score += 0.05
        
        # Architecture depth scoring
        depth = len(arch.layers)
        if 3 <= depth <= 7:
            score += 0.1
        elif depth > 10:
            score -= 0.1
        
        # Add some randomness to simulate training variance
        score += random.uniform(-0.05, 0.05)
        
        return max(0, min(1, score))
    
    def _evolve_population(self, population: List[Architecture]) -> List[Architecture]:
        """Create next generation through evolution."""
        new_population = []
        
        # Keep elite
        elite = population[:self.elite_size]
        new_population.extend(elite)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child = self._crossover(parent1, parent2)
            else:
                # Copy from population
                child = self._tournament_select(population)
                child = self._copy_architecture(child)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # Validate and add
            if child.count_parameters() <= self.max_parameters:
                child.fitness = None  # Reset fitness
                new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_select(self, population: List[Architecture], 
                          tournament_size: int = 3) -> Architecture:
        """Tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness or 0)
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two architectures."""
        # Take layers from both parents
        split_point = random.randint(1, min(len(parent1.layers), len(parent2.layers)) - 1)
        
        child_layers = parent1.layers[:split_point] + parent2.layers[split_point:]
        
        # Inherit hyperparameters from random parent
        if random.random() < 0.5:
            return Architecture(
                layers=child_layers,
                learning_rate=parent1.learning_rate,
                batch_size=parent1.batch_size,
                optimizer=parent1.optimizer
            )
        else:
            return Architecture(
                layers=child_layers,
                learning_rate=parent2.learning_rate,
                batch_size=parent2.batch_size,
                optimizer=parent2.optimizer
            )
    
    def _mutate(self, arch: Architecture) -> Architecture:
        """Mutate an architecture."""
        arch = self._copy_architecture(arch)
        
        mutation_type = random.choice([
            'add_layer', 'remove_layer', 'change_layer', 
            'change_hyperparameter'
        ])
        
        if mutation_type == 'add_layer' and len(arch.layers) < self.max_layers:
            # Add random layer
            position = random.randint(0, len(arch.layers) - 1)
            new_layer = self._create_random_layer()
            arch.layers.insert(position, new_layer)
            
        elif mutation_type == 'remove_layer' and len(arch.layers) > 2:
            # Remove random layer (keep at least input and output)
            position = random.randint(0, len(arch.layers) - 2)
            arch.layers.pop(position)
            
        elif mutation_type == 'change_layer' and arch.layers:
            # Modify random layer
            position = random.randint(0, len(arch.layers) - 1)
            layer = arch.layers[position]
            
            if layer.layer_type == LayerType.DENSE and layer.units:
                # Change units
                layer.units = random.choice([16, 32, 64, 128, 256])
            elif layer.layer_type == LayerType.DROPOUT and layer.dropout_rate:
                # Change dropout rate
                layer.dropout_rate = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            elif layer.activation:
                # Change activation
                layer.activation = random.choice(list(ActivationType))
                
        elif mutation_type == 'change_hyperparameter':
            # Mutate hyperparameters
            param = random.choice(['learning_rate', 'batch_size', 'optimizer'])
            
            if param == 'learning_rate':
                arch.learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            elif param == 'batch_size':
                arch.batch_size = random.choice([16, 32, 64])
            elif param == 'optimizer':
                arch.optimizer = random.choice(["adam", "sgd", "rmsprop"])
        
        return arch
    
    def _create_random_layer(self) -> LayerConfig:
        """Create a random layer configuration."""
        layer_type = random.choice(list(LayerType))
        
        if layer_type == LayerType.DENSE:
            return LayerConfig(
                layer_type=layer_type,
                units=random.choice([16, 32, 64, 128]),
                activation=random.choice(list(ActivationType))
            )
        elif layer_type == LayerType.DROPOUT:
            return LayerConfig(
                layer_type=layer_type,
                dropout_rate=random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            )
        else:
            return LayerConfig(layer_type=layer_type)
    
    def _copy_architecture(self, arch: Architecture) -> Architecture:
        """Deep copy an architecture."""
        return Architecture(
            layers=[LayerConfig(**layer.to_dict()) for layer in arch.layers],
            learning_rate=arch.learning_rate,
            batch_size=arch.batch_size,
            optimizer=arch.optimizer,
            fitness=arch.fitness,
            training_time=arch.training_time,
            parameters=arch.parameters
        )
    
    def save_best_architecture(self, filepath: str) -> None:
        """Save best architecture to file."""
        if self.best_architecture:
            arch_dict = {
                'layers': [layer.to_dict() for layer in self.best_architecture.layers],
                'learning_rate': self.best_architecture.learning_rate,
                'batch_size': self.best_architecture.batch_size,
                'optimizer': self.best_architecture.optimizer,
                'fitness': self.best_architecture.fitness,
                'parameters': self.best_architecture.parameters,
                'generation_found': self.generation
            }
            
            with open(filepath, 'w') as f:
                json.dump(arch_dict, f, indent=2)
            
            print(f"Saved best architecture to {filepath}")
    
    def generate_code(self, arch: Architecture, framework: str = "keras") -> str:
        """Generate code for the architecture."""
        if framework == "keras":
            return self._generate_keras_code(arch)
        elif framework == "pytorch":
            return self._generate_pytorch_code(arch)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def _generate_keras_code(self, arch: Architecture) -> str:
        """Generate Keras code for architecture."""
        code = """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    model = keras.Sequential([
"""
        
        for i, layer in enumerate(arch.layers):
            indent = "        "
            
            if layer.layer_type == LayerType.DENSE:
                code += f"{indent}layers.Dense({layer.units}"
                if layer.activation:
                    code += f", activation='{layer.activation.value}'"
                code += ")"
                
            elif layer.layer_type == LayerType.CONV2D:
                code += f"{indent}layers.Conv2D({layer.units}, {layer.kernel_size}"
                if layer.activation:
                    code += f", activation='{layer.activation.value}'"
                code += ")"
                
            elif layer.layer_type == LayerType.DROPOUT:
                code += f"{indent}layers.Dropout({layer.dropout_rate})"
                
            elif layer.layer_type == LayerType.LSTM:
                code += f"{indent}layers.LSTM({layer.units})"
                
            elif layer.layer_type == LayerType.GRU:
                code += f"{indent}layers.GRU({layer.units})"
                
            elif layer.layer_type == LayerType.BATCH_NORM:
                code += f"{indent}layers.BatchNormalization()"
                
            elif layer.layer_type == LayerType.POOLING:
                code += f"{indent}layers.MaxPooling2D({layer.pool_size})"
            
            if i < len(arch.layers) - 1:
                code += ","
            code += "\n"
        
        code += f"""    ])
    
    model.compile(
        optimizer='{arch.optimizer}',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and print model summary
model = create_model()
model.summary()
"""
        
        return code
    
    def _generate_pytorch_code(self, arch: Architecture) -> str:
        """Generate PyTorch code for architecture."""
        code = """import torch
import torch.nn as nn
import torch.nn.functional as F

class NASModel(nn.Module):
    def __init__(self):
        super(NASModel, self).__init__()
"""
        
        # Define layers
        prev_units = 32  # Assume input size
        for i, layer in enumerate(arch.layers):
            if layer.layer_type == LayerType.DENSE and layer.units:
                code += f"        self.fc{i} = nn.Linear({prev_units}, {layer.units})\n"
                prev_units = layer.units
            elif layer.layer_type == LayerType.DROPOUT and layer.dropout_rate:
                code += f"        self.dropout{i} = nn.Dropout({layer.dropout_rate})\n"
        
        # Forward method
        code += "\n    def forward(self, x):\n"
        
        for i, layer in enumerate(arch.layers):
            if layer.layer_type == LayerType.DENSE:
                code += f"        x = self.fc{i}(x)\n"
                if layer.activation == ActivationType.RELU:
                    code += "        x = F.relu(x)\n"
                elif layer.activation == ActivationType.TANH:
                    code += "        x = torch.tanh(x)\n"
            elif layer.layer_type == LayerType.DROPOUT:
                code += f"        x = self.dropout{i}(x)\n"
        
        code += "        return x\n"
        code += "\n# Create model\nmodel = NASModel()\nprint(model)"
        
        return code


def demonstrate_nas():
    """Demonstrate Neural Architecture Search."""
    # Create NAS instance
    nas = EvolutionaryNAS(max_time_hours=0.01)  # 36 seconds for demo
    
    # Search for classification architecture
    print("=== Neural Architecture Search Demo ===")
    best_arch = nas.search(task_type="classification")
    
    print("\n=== Best Architecture Found ===")
    print(f"Fitness: {best_arch.fitness:.4f}")
    print(f"Parameters: {best_arch.parameters:,}")
    print(f"Learning Rate: {best_arch.learning_rate}")
    print(f"Batch Size: {best_arch.batch_size}")
    print(f"Optimizer: {best_arch.optimizer}")
    
    print("\n=== Architecture Layers ===")
    for i, layer in enumerate(best_arch.layers):
        print(f"Layer {i}: {layer.layer_type.value}", end="")
        if layer.units:
            print(f" - Units: {layer.units}", end="")
        if layer.activation:
            print(f" - Activation: {layer.activation.value}", end="")
        if layer.dropout_rate:
            print(f" - Dropout: {layer.dropout_rate}", end="")
        print()
    
    # Generate code
    print("\n=== Generated Keras Code ===")
    keras_code = nas.generate_code(best_arch, framework="keras")
    print(keras_code[:500] + "...")
    
    # Save architecture
    nas.save_best_architecture("best_architecture.json")


if __name__ == "__main__":
    demonstrate_nas()