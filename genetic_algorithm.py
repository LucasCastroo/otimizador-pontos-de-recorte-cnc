from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(
        self,
        TAM_POP: int,
        recortes_disponiveis: List[Dict[str, Any]],
        sheet_width: float,
        sheet_height: float,
        numero_geracoes: int = 100
    ):

        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_layout: List[Dict[str, Any]] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        self.mutation_rate = 0.1
        self.elitism = True

        self.initialize_population()


    def initialize_population(self):
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)


    def decode_layout(self, permutation: List[int]) -> Tuple[List[Dict[str,Any]], int]:
        layout_result: List[Dict[str, Any]] = []
        free_rects: List[Tuple[float,float,float,float]] = []

        free_rects.append((0, 0, self.sheet_width, self.sheet_height))

        discarded = 0

        for idx in permutation:
            rec = self.recortes_disponiveis[idx]
            possible_configs = []
            if rec["tipo"] in ("retangular","diamante"):
                for rot in [0, 90]:
                    w,h = self.get_dims(rec, rot)
                    possible_configs.append((rot, w, h))
            else:
                w,h = self.get_dims(rec, 0)
                possible_configs.append((0, w, h))

            placed = False
            for (rot, w, h) in possible_configs:
                best_index = -1
                for i, (rx, ry, rw, rh) in enumerate(free_rects):
                    if w <= rw and h <= rh:
                        best_index = i
                        break
                if best_index != -1:
                    placed = True
                    r_final = copy.deepcopy(rec)
                    r_final["rotacao"] = rot
                    (rx, ry, rw, rh) = free_rects[best_index]
                    r_final["x"] = rx
                    r_final["y"] = ry
                    layout_result.append(r_final)

                    del free_rects[best_index]

                    if w < rw:
                        newW = rw - w
                        if newW > 0:
                            free_rects.append((rx + w, ry, newW, rh))
                    if h < rh:
                        newH = rh - h
                        if newH > 0:
                            free_rects.append((rx, ry + h, w, newH))
                    break
            if not placed:
                discarded += 1

        return (layout_result, discarded)


    def get_dims(self, rec: Dict[str,Any], rot: int) -> Tuple[float,float]:
        tipo = rec["tipo"]
        if tipo == "circular":
            d = 2*rec["r"]
            return (d, d)
        elif tipo in ("retangular","diamante"):
            if rot == 90:
                return (rec["altura"], rec["largura"])
            else:
                return (rec["largura"], rec["altura"])
        else:
            return (rec.get("largura",10), rec.get("altura",10))


    def evaluate_individual(self, permutation: List[int]) -> float:
        layout, discarded = self.decode_layout(permutation)

        if not layout:
            return self.sheet_width*self.sheet_height*2 + discarded*10000

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        for rec in layout:
            angle = rec.get("rotacao", 0)
            w,h = self.get_dims(rec, angle)
            x0, y0 = rec["x"], rec["y"]
            x1, y1 = x0 + w, y0 + h
            x_min = min(x_min, x0)
            x_max = max(x_max, x1)
            y_min = min(y_min, y0)
            y_max = max(y_max, y1)

        area_layout = (x_max - x_min)*(y_max - y_min)
        penalty = discarded*10000
        return area_layout + penalty


    def evaluate_population(self):
        for perm in self.POP:
            fit = self.evaluate_individual(perm)
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_individual = perm[:]


    def compute_fitness_scores(self) -> List[float]:
        fits = [self.evaluate_individual(perm) for perm in self.POP]
        return [1/(1+f) for f in fits]
    

    def roulette_selection(self) -> List[int]:
        scores = self.compute_fitness_scores()
        total = sum(scores)
        pick = random.random()*total
        current=0
        for perm, sc in zip(self.POP, scores):
            current+=sc
            if current>=pick:
                return perm
        return self.POP[-1]


    def crossover_two_point(self, p1: List[int], p2: List[int]) -> List[int]:
        size = len(p1)
        i1, i2 = sorted(random.sample(range(size),2))
        child = [None]*size
        child[i1:i2+1] = p1[i1:i2+1]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx+=1
                child[i] = p2[p2_idx]
                p2_idx+=1
        return child


    def mutate(self, perm: List[int]) -> List[int]:
        if random.random()<self.mutation_rate:
            i1,i2 = random.sample(range(len(perm)),2)
            perm[i1], perm[i2] = perm[i2], perm[i1]
        return perm

    def genetic_operators(self):
        new_pop = []
        if self.elitism and self.best_individual:
            new_pop.append(self.best_individual[:])
        while len(new_pop)<self.TAM_POP:
            p1 = self.roulette_selection()
            p2 = self.roulette_selection()
            child = self.crossover_two_point(p1,p2)
            child = self.mutate(child)
            new_pop.append(child)
        self.POP = new_pop[:self.TAM_POP]


    def run(self):
        for gen in range(self.numero_geracoes):
            self.evaluate_population()
            self.genetic_operators()
            if gen%20==0:
                print(f"Generation {gen} -- Fitness = {self.best_fitness}")
        layout, discarded = self.decode_layout(self.best_individual)
        self.optimized_layout = layout
        return self.optimized_layout


    def optimize_and_display(self):

        self.display_layout(self.recortes_disponiveis, title="Initial Layout - Genetic Algorithm - Lucas Castro")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm - Lucas Castro")
        return self.optimized_layout