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

        """
        Inicializa o Algoritmo Genético para otimização do corte de chapas.

        Parâmetros:
        - TAM_POP: Tamanho da população (quantidade de indivíduos)
        - recortes_disponiveis: Lista contendo os recortes a serem organizados
        - sheet_width: Largura da chapa
        - sheet_height: Altura da chapa
        - numero_geracoes: Número total de gerações para a evolução
        """
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # População e melhor indivíduo encontrado
        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_layout: List[Dict[str, Any]] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        # Parâmetros do Algoritmo Genético
        self.mutation_rate = 0.1 # Probabilidade de mutação
        self.elitism = True # Mantém o melhor indivíduo na próxima geração

        self.initialize_population()


    def initialize_population(self):
        """
        Inicializa a população com sequências aleatórias das peças disponíveis.
        Cada indivíduo é uma permutação dos índices dos recortes.
        """
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)


    def decode_layout(self, permutation: List[int]) -> Tuple[List[Dict[str,Any]], int]:
        """
        Converte uma permutação de peças em um layout organizado na chapa.

        - Usa o método **Retângulos Livres** para alocar as peças.
        - Se a peça não couber, ela é descartada e penalizada.

        Retorna:
        - layout_result: Lista com as peças posicionadas
        - discarded: Número de peças que não couberam na chapa
        """
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
        """
        Retorna as dimensões de um recorte considerando a rotação.
        """
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
        """
        Calcula o fitness de um indivíduo:
        - Minimiza a área ocupada.
        - Penaliza peças descartadas.
        """
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
        """
        Avalia a aptidão (fitness) de todos os indivíduos na população atual.

        - Percorre cada indivíduo da população e calcula seu fitness.
        - Atualiza o melhor indivíduo encontrado e seu respectivo fitness.

        A função mantém um acompanhamento contínuo do melhor layout encontrado
        até o momento, garantindo que a otimização continue evoluindo para melhores soluções.
        """
        for perm in self.POP:
            fit = self.evaluate_individual(perm)
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_individual = perm[:]


    def compute_fitness_scores(self) -> List[float]:
        """
        Calcula os scores de fitness para cada indivíduo da população.

        - Utiliza a métrica de aptidão de cada indivíduo (`evaluate_individual`).
        - Converte o valor de fitness para uma escala positiva onde **menores valores** significam melhor aptidão.
        - O cálculo `1/(1+f)` garante que indivíduos com fitness menor tenham maior peso na seleção.

        Retorna:
        - Uma lista de scores de fitness, usada para a **seleção por roleta**.
        """
        fits = [self.evaluate_individual(perm) for perm in self.POP]
        return [1/(1+f) for f in fits]
    

    def roulette_selection(self) -> List[int]:
        """
        Realiza a **seleção por roleta** para escolher um indivíduo da população.

        - A probabilidade de um indivíduo ser escolhido é proporcional ao seu fitness.
        - Indivíduos com **menor fitness** (melhor solução) têm **maior chance** de serem escolhidos.

        Funcionamento:
        1. Soma os scores de fitness de todos os indivíduos.
        2. Gera um número aleatório no intervalo da soma total.
        3. Percorre os indivíduos e seleciona o primeiro cuja soma cumulativa dos scores atinja o valor aleatório.

        Retorna:
        - Um indivíduo selecionado para ser usado no cruzamento (crossover).
        """
        scores = self.compute_fitness_scores()
        total = sum(scores)
        pick = random.random()*total # Valor aleatório no intervalo do total dos scores
        current=0
        for perm, sc in zip(self.POP, scores):
            current+=sc
            if current>=pick:
                return perm
        return self.POP[-1] # Retorna o último caso nenhum seja escolhido


    def crossover_two_point(self, p1: List[int], p2: List[int]) -> List[int]:
        """
        Aplica **crossover de dois pontos** entre dois indivíduos.

        - O crossover cria um novo indivíduo combinando partes de dois pais.
        - Escolhe dois pontos aleatórios no vetor do indivíduo.
        - A parte intermediária de **p1** é mantida e o restante é preenchido com os elementos de **p2** na mesma ordem.

        Funcionamento:
        1. Escolhe dois índices aleatórios dentro do tamanho do vetor.
        2. Mantém os elementos entre esses dois pontos do pai `p1`.
        3. Preenche os espaços restantes com os genes de `p2`, mantendo a ordem original.

        Retorna:
        - Um novo indivíduo resultante do cruzamento.

        Exemplo:
        ```
        p1 = [1, 2, 3, 4, 5, 6, 7, 8]
        p2 = [3, 7, 5, 1, 6, 8, 2, 4]
        
        # Suponha que os pontos escolhidos sejam i1=2 e i2=5
        # O filho mantém os elementos 3,4,5,6 de p1 e preenche o restante com p2 mantendo a ordem.

        Filho gerado -> [7, 3, 3, 4, 5, 6, 1, 8]
        ```
        """
        size = len(p1)
        i1, i2 = sorted(random.sample(range(size),2)) # Escolhe dois pontos aleatórios
        child = [None]*size

        # Copia a fatia de p1 para o filho
        child[i1:i2+1] = p1[i1:i2+1]

        p2_idx = 0
        for i in range(size):
            if child[i] is None: # Preenche os espaços vazios com elementos de p2
                while p2[p2_idx] in child:
                    p2_idx+=1
                child[i] = p2[p2_idx]
                p2_idx+=1
        return child


    def mutate(self, perm: List[int]) -> List[int]:
        """
        Aplica mutação a um indivíduo.

        - Troca aleatoriamente a posição de dois elementos dentro do vetor do indivíduo.
        - A mutação ocorre com uma probabilidade definida (`self.mutation_rate`).
        - A troca de genes ajuda a introduzir variação na população, evitando **convergência prematura**.

        Funcionamento:
        1. Sorteia dois índices aleatórios no vetor.
        2. Troca os elementos dessas posições.

        Retorna:
        - O indivíduo após a mutação.

        Exemplo:
        ```
        Antes da mutação: [1, 2, 3, 4, 5]
        Índices sorteados: i1 = 1, i2 = 3
        Depois da mutação: [1, 4, 3, 2, 5]
        ```
        """
        if random.random()<self.mutation_rate: # Executa mutação com certa probabilidade
            i1,i2 = random.sample(range(len(perm)),2) # Seleciona dois índices aleatórios
            perm[i1], perm[i2] = perm[i2], perm[i1] # Troca os elementos
        return perm

    def genetic_operators(self):
        """
        Aplica os operadores genéticos de crossover e mutação na população.
        """
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
        """
        Executa o algoritmo por `N` gerações e imprime progresso no terminal.
        """
        print("\n---------------------------------------------------")
        print("   GERAÇÃO   |   FITNESS")
        print("---------------------------------------------------")
        for gen in range(self.numero_geracoes):
            self.evaluate_population()
            self.genetic_operators()
            if gen%20==0:
                print(f"🌀   {gen}      |   {self.best_fitness}")
        print("---------------------------------------------------")
        layout, discarded = self.decode_layout(self.best_individual)
        self.optimized_layout = layout
        return self.optimized_layout


    def optimize_and_display(self):
        """
        Exibe o layout inicial e o layout otimizado após a execução.
        """
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - Genetic Algorithm - Lucas Castro")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm - Lucas Castro")
        return self.optimized_layout