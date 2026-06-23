import numpy as np

class BenfordAnalyzer:
    @staticmethod
    def leading_digit_distribution(values):
        # Flatten e remove zeros
        values = np.array(values).flatten()
        values = values[values > 0]

        # Extrai o primeiro dígito
        leading_digits = [int(str(int(v))[0]) for v in values if v > 0]

        # Conta frequência relativa
        counts = np.zeros(9)
        for d in leading_digits:
            if 1 <= d <= 9:
                counts[d-1] += 1

        return counts / np.sum(counts)

    @staticmethod
    def benford_expected():
        # Distribuição teórica da Lei de Benford
        return np.array([np.log10(1 + 1/d) for d in range(1, 10)])

    @staticmethod
    def benford_score(values):
        observed = BenfordAnalyzer.leading_digit_distribution(values)
        expected = BenfordAnalyzer.benford_expected()
        # Distância simples (quanto menor, mais próximo da lei)
        return np.linalg.norm(observed - expected)
