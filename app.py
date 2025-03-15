from ant_colony import AntColony
from differential_evolution import DifferentialEvolution
from genetic_algorithm import GeneticAlgorithm
from particle_swarm import ParticleSwarm
import time

def main():
    # Define sheet dimensions
    sheet_width = 220
    sheet_height = 120

    # Define available parts (recortes_disponiveis) as a JSON-like structure.
    # Examples of one of each type of part:
    # recortes_disponiveis = [
    #     {"tipo": "retangular", "largura": 20, "altura": 10, "x": 0, "y": 0, "rotacao": 0},
    #     {"tipo": "circular", "r": 10, "x": 0, "y": 0},
    #     {"tipo": "triangular", "b": 25, "h": 20, "x": 0, "y": 0, "rotacao": 10},
    #     {"tipo": "diamante", "largura": 30, "altura": 20, "x": 0, "y": 0, "rotacao": 0}
    # ]

    recortes_disponiveis = [
        {"tipo": "retangular", "largura": 35, "altura": 20, "x": 1, "y": 1, "rotacao": 0},
        {"tipo": "retangular", "largura": 40, "altura": 25, "x": 30, "y": 1, "rotacao": 0},
        {"tipo": "diamante", "largura": 30, "altura": 50, "x": 70, "y": 2, "rotacao": 0},
        {"tipo": "retangular", "largura": 60, "altura": 20, "x": 10, "y": 40, "rotacao": 0},
        {"tipo": "retangular", "largura": 50, "altura": 15, "x": 15, "y": 80, "rotacao": 0},
        {"tipo": "circular", "r": 18, "x": 140, "y": 10},
        {"tipo": "circular", "r": 16, "x": 170, "y": 15},
        {"tipo": "diamante", "largura": 40, "altura": 45, "x": 110, "y": 40, "rotacao": 0},
        {"tipo": "retangular", "largura": 90, "altura": 30, "x": 50, "y": 70, "rotacao": 0}
    ]

    """
    # Instantiate and run Ant Colony Optimization.
    ant_optimizer = AntColony(num_ants=50, num_iterations=100, sheet_width=sheet_width,
                              sheet_height=sheet_height, recortes_disponiveis=recortes_disponiveis)
    print("Running Ant Colony Optimization...")
    ant_optimized_layout = ant_optimizer.optimize_and_display()
    """

    """
    # Instantiate and run Differential Evolution.
    de_optimizer = DifferentialEvolution(pop_size=50, max_iter=100, sheet_width=sheet_width,
                                         sheet_height=sheet_height, recortes_disponiveis=recortes_disponiveis)
    print("Running Differential Evolution...")
    de_optimized_layout = de_optimizer.optimize_and_display()
    """

    # Instantiate and run Genetic Algorithm.
    print("\n🟢 Iniciando Algoritmo Genético...")
    start_time = time.time()

    ga_optimizer = GeneticAlgorithm(TAM_POP=50, recortes_disponiveis=recortes_disponiveis,
                                    sheet_width=sheet_width, sheet_height=sheet_height, numero_geracoes=100)
    
    print("⚙️  Executando otimização...")
    ga_optimized_layout = ga_optimizer.optimize_and_display()

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n✅ Otimização Concluída!")
    print(f"⏱️ Tempo total de execução: {execution_time:.2f} segundos")
    print(f"🏆 Melhor Fitness Obtido: {ga_optimizer.best_fitness:.2f}")

    """
    # Instantiate and run Particle Swarm Optimization.
    ps_optimizer = ParticleSwarm(num_particles=50, num_iterations=100, dim=len(recortes_disponiveis),
                                 sheet_width=sheet_width, sheet_height=sheet_height, recortes_disponiveis=recortes_disponiveis)
    print("Running Particle Swarm Optimization...")
    ps_optimized_layout = ps_optimizer.optimize_and_display()"
    """

if __name__ == "__main__":
    main()
