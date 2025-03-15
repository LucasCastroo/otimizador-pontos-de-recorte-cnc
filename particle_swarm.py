from common.layout_display import LayoutDisplayMixin
import numpy as np

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.recortes_disponiveis = recortes_disponiveis

        self.particles = self.initialize_particles()
        self.velocities = np.zeros((num_particles, dim, 2))

        self.p_best = np.copy(self.particles)
        self.p_best_scores = np.full(num_particles, np.inf)

        self.g_best = np.copy(self.particles[0])
        self.g_best_score = self.evaluate(self.particles[0])

    def initialize_particles(self):
        particles = np.zeros((self.num_particles, self.dim, 2))
        rect_x, rect_y = 5, 5
        circle_x, circle_y = self.sheet_width // 3, 10
        diamond_x, diamond_y = self.sheet_width // 2, self.sheet_height - 40

        for i in range(self.num_particles):
            for j, recorte in enumerate(self.recortes_disponiveis):
                if recorte["tipo"] == "retangular":
                    x, y = rect_x, rect_y
                    rect_x += recorte["largura"] + 10
                    if rect_x + recorte["largura"] > self.sheet_width // 2:
                        rect_x = 5
                        rect_y += recorte["altura"] + 10

                elif recorte["tipo"] == "circular":
                    x, y = circle_x, circle_y
                    circle_y += recorte["r"] * 2 + 12

                elif recorte["tipo"] == "diamante":
                    x, y = diamond_x, diamond_y
                    diamond_x += recorte["largura"] + 10
                    if diamond_x + recorte["largura"] > self.sheet_width - 5:
                        diamond_x = self.sheet_width // 2
                        diamond_y -= recorte["altura"] + 10

                particles[i, j] = [x, y]

        return particles

    def evaluate(self, positions):
        total_area = self.sheet_width * self.sheet_height
        used_area = 0
        occupied = np.zeros((self.sheet_width, self.sheet_height))
        overlap_penalty = 0
        spacing_penalty = 0

        for i, recorte in enumerate(self.recortes_disponiveis):
            x, y = int(positions[i][0]), int(positions[i][1])

            if recorte["tipo"] == "retangular":
                largura, altura = recorte["largura"], recorte["altura"]
            elif recorte["tipo"] == "circular":
                largura, altura = 2 * recorte["r"], 2 * recorte["r"]
            elif recorte["tipo"] == "diamante":
                largura, altura = recorte["largura"], recorte["altura"]
            else:
                largura, altura = 0, 0

            if x + largura > self.sheet_width or y + altura > self.sheet_height or x < 0 or y < 0:
                return np.inf

            if np.sum(occupied[x:x+largura, y:y+altura]) > 0:
                overlap_penalty += 20000  

            for j, other in enumerate(self.recortes_disponiveis):
                if i != j:
                    x2, y2 = int(positions[j][0]), int(positions[j][1])
                    distancia = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                    if distancia < 20:  
                        spacing_penalty += 4000

            occupied[x:x+largura, y:y+altura] = 1
            used_area += largura * altura

        wasted_space = total_area - used_area
        return wasted_space + overlap_penalty + spacing_penalty

    def update_particles(self):
        inertia = 0.3  
        cognitive = 1.2  
        social = 1.4  

        for i in range(self.num_particles):
            if self.g_best is None or np.isinf(self.g_best_score):
                continue

            social_component = social * np.random.rand() * (self.g_best - self.particles[i])

            self.velocities[i] = (
                inertia * self.velocities[i] +
                cognitive * np.random.rand() * (self.p_best[i] - self.particles[i]) +
                social_component
            )

            self.velocities[i] = np.clip(self.velocities[i], -1.0, 1.0)

            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], 0, [self.sheet_width, self.sheet_height])

    def run(self):
        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                score = self.evaluate(self.particles[i])

                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best[i] = self.particles[i]

                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best = np.copy(self.particles[i])

            print(f"Iteração {iteration + 1}/{self.num_iterations} - Melhor Score: {self.g_best_score}")

            self.update_particles()

        optimized_layout = []
        for i, recorte in enumerate(self.recortes_disponiveis):
            optimized_layout.append({
                "tipo": recorte["tipo"],
                "x": int(self.g_best[i][0]),
                "y": int(self.g_best[i][1]),
                "largura": recorte.get("largura", 0),
                "altura": recorte.get("altura", 0),
                "r": recorte.get("r", 0),
                "rotacao": recorte.get("rotacao", 0)
            })

        return optimized_layout

    def optimize_and_display(self):
        print("Iniciando otimização com PSO...")
        self.optimized_layout = self.run()

        print("Exibindo layout otimizado...")
        self.display_layout(self.optimized_layout, title="Optimized Layout - Particle Swarm")

        return self.optimized_layout
