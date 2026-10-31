"""Gradient check: confere o backprop contra gradientes numéricos.

A ideia: o backprop calcula o gradiente de forma analítica (regra da cadeia).
Dá para conferir cada entrada do gradiente perturbando o parâmetro e medindo
como a loss muda (diferenças finitas). Se os dois valores baterem, o backprop
está certo.

Usa diferenças centrais, ``(f(x + eps) - f(x - eps)) / (2 * eps)``, que têm
erro de ordem ``eps^2``, bem menor que o das diferenças progressivas. Tudo em
float64, porque em float32 o ruído numérico esconde o sinal.
"""

from __future__ import annotations

import numpy as np

from src.network import MLP


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Erro relativo entre o gradiente analítico e o numérico.

    Fórmula: ``‖a - n‖ / (‖a‖ + ‖n‖)``, com ``a`` analítico e ``n`` numérico.
    O termo ``1e-12`` evita divisão por zero quando ambos são nulos.
    """
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    numerator = np.linalg.norm(analytic - numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
    return float(numerator / denominator)


def numerical_gradient(
    loss_at: callable, parameter: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Gradiente numérico de ``parameter`` por diferenças centrais.

    Args:
        loss_at: função sem argumentos que recalcula a loss com os valores
            atuais dos parâmetros.
        parameter: array que será perturbado, uma entrada por vez. É
            restaurado ao valor original ao fim de cada entrada.
        epsilon: tamanho da perturbação.

    Returns:
        Array do mesmo shape de ``parameter`` com o gradiente numérico.
    """
    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + epsilon
        loss_plus = loss_at()
        parameter[index] = original - epsilon
        loss_minus = loss_at()
        parameter[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient


def check_mlp(
    layer_sizes: tuple[int, ...] = (6, 8, 5, 3),
    batch_size: int = 4,
    seed: int = 0,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Roda o gradient check numa MLP e devolve o erro relativo por parâmetro.

    Returns:
        Dicionário ``{"linear_0.weights": erro, "linear_0.bias": erro, ...}``.
    """
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(batch_size, layer_sizes[0]))
    targets = rng.integers(0, layer_sizes[-1], size=batch_size)

    network = MLP(layer_sizes, seed=seed)

    def loss_at() -> float:
        return network.forward(inputs, targets)

    # Forward e backward uma vez para obter os gradientes analíticos.
    loss_at()
    network.backward()

    errors: dict[str, float] = {}
    for position, (parameter, analytic_gradient) in enumerate(network.parameters()):
        layer_index, kind = divmod(position, 2)
        name = f"linear_{layer_index}.{'weights' if kind == 0 else 'bias'}"
        numeric_gradient = numerical_gradient(loss_at, parameter, epsilon)
        errors[name] = relative_error(analytic_gradient, numeric_gradient)
    return errors


def main() -> None:
    errors = check_mlp()
    worst = max(errors.values())
    for name, error in errors.items():
        print(f"{name:20s} erro relativo = {error:.3e}")
    print(f"\npior caso: {worst:.3e} (limite: 1e-7)")
    if worst < 1e-7:
        print("OK: backprop confere com o gradiente numérico")
    else:
        raise SystemExit("FALHOU: algum gradiente ficou acima do limite")


if __name__ == "__main__":
    main()
