import numpy as np
from sympy import *
from typing import Any, List, Tuple


def pente_extreme(x: np.ndarray, y: np.ndarray, dy: float = 0, dx: float = 0) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calcul de la pente d'une fonction avec une erreur sur la pente.
    Retourne le coefficient directeur de la pente et son incertitude, ainsi que les points de la pente avec coefficient
    max et min.
    :param x: Abscisses de la fonction
    :param y: Ordonnees de la fonction
    :param dy: Erreur sur les ordonnees de la fonction
    :param dx: Erreur sur les abscisses de la fonction
    :return: Coefficient directeur de la pente, incertitude sur le coefficient directeur, points de la pente avec
    coefficient
    """
    a_max = (y[-1] + dy - (y[0] - dy)) / (x[-1] - dx - x[0] - dx)
    a_min = (y[-1] - dy - (y[0] + dy)) / (x[-1] + dx - x[0] + dx)

    a = (a_max + a_min) / 2
    delta_a = (a_max - a_min) / 2

    return a, delta_a, a_max * x + y[0] - dy, a_min * x + y[0] + dy


def incertitude_derivee_partielle(symbols: List[Symbol], symbols_value: List[float], uncertainties: List[float],
                                  expression) -> Tuple[float, float]:
    """
    Calcul de l'incertitude par la méthode des dérivées partielles
    :param symbols: Liste des symboles de l'expression
    :param symbols_value: Valeur de chaque grandeur que l'on dérive
    :param uncertainties: Liste des incertitudes sur les symboles de l'expression
    :param expression: Expression dont on veut calculer l'incertitude sur la derivee partielle
    :return: Valeur de l'expression puis l'incertitude sur la derivee partielle de l'expression
    """
    init_printing()
    if len(symbols) != len(uncertainties) or len(symbols) != len(symbols_value):
        raise Exception("Il faut autant de symboles que d'incertitudes associées !")

    # On calcule la derivee partielle de l'expression selon chaque symbole
    expression_uncertainty: float = 0
    expression_value: float = expression.subs(zip(symbols, symbols_value))
    for symbol, uncertainty in zip(symbols, uncertainties):
        derivative = diff(expression, symbol)
        derivative_value = derivative.subs(zip(symbols, symbols_value))
        expression_uncertainty += abs(derivative_value) * uncertainty

    return expression_value, expression_uncertainty
