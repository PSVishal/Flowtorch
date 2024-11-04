"""Implementation of a linear dynamics model with control inputs.
"""

from typing import Tuple
from numpy import pi
import torch as pt
from .dmd import _dft_properties


def _unsqueeze_if_1d(tensor: pt.Tensor, last: bool = False):
    dim = 1 if last else 0
    return tensor.unsqueeze(dim=dim) if len(tensor.shape) < 2 else tensor


class LinearControl(object):
    def __init__(self, data_matrix: pt.Tensor, control_inputs: pt.Tensor, dt: float):
        self._dm = data_matrix
        self._n_states, self._n_times = self._dm.shape
        self._cm = _unsqueeze_if_1d(control_inputs)
        self._n_controls = self._cm.shape[0]
        self._dt = dt
        self._check_input_consistency()
        self._A, self._B = self._compute_operators()
        self._eigvals, self._eigvecs = pt.linalg.eig(self._A)
        self._amplitude = pt.linalg.pinv(self._eigvecs) @ self._dm[:, 0].type(
            self._eigvecs.dtype
        )

    def _check_input_consistency(self):
        control_steps = self._cm.shape[1]
        if not control_steps + 1 == self._n_times:
            raise ValueError(
                "The number of control inputs (n_controls) should be\n"
                + "one less than the number of state vectors (n_states).\n"
                + f"Got n_states={self._n_times:d} and n_controls={control_steps:d}"
            )

    def _compute_operators(self) -> Tuple[pt.Tensor, pt.Tensor]:
        X, Y, G = self._dm[:, :-1], self._dm[:, 1:], self._cm
        AB = Y @ pt.linalg.pinv(pt.cat((X, G), dim=0))
        return AB[:, : self._n_states], AB[:, self._n_states :]

    def predict(
        self, initial_condition: pt.Tensor, control_inputs: pt.Tensor
    ) -> pt.Tensor:
        cm = _unsqueeze_if_1d(control_inputs)
        if not cm.shape[0] == self._n_controls:
            raise ValueError(
                f"Expected {self._n_controls:d} control inputs but got {cm.shape[0]:d}"
            )
        b = pt.linalg.pinv(self._eigvecs) @ initial_condition.type(self._eigvecs.dtype)
        n_steps = cm.shape[-1] + 1
        V = pt.linalg.vander(self._eigvals, N=n_steps)
        C = pt.linalg.pinv(self._eigvecs) @ (self._B @ cm).type(self._eigvecs.dtype)
        forcing = pt.vstack(
            [(C[:, :n] * V[:, :n].flip(-1)).sum(dim=1) for n in range(1, n_steps)]
        ).T
        forcing = pt.cat((pt.zeros(self._n_states).unsqueeze(-1), forcing), dim=1)
        return (self._eigvecs @ (V * b.unsqueeze(-1) + forcing)).type(self._dm.dtype)

    @property
    def A(self) -> pt.Tensor:
        return self._A

    @property
    def B(self) -> pt.Tensor:
        return self._B

    @property
    def eigvals(self) -> pt.Tensor:
        return self._eigvals

    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self._eigvals) / self._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        return self._eigvecs

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self._eigvals).imag / (2.0 * pi * self._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self._eigvals) / self._dt).real

    @property
    def amplitude(self) -> pt.Tensor:
        return self._amplitude

    @property
    def modes(self) -> pt.Tensor:
        return self.eigvecs

    @property
    def unforced_dynamics(self) -> pt.Tensor:
        return pt.linalg.vander(
            self._eigvals, N=self._n_times
        ) * self._amplitude.unsqueeze(-1)

    @property
    def forced_dynamics(self) -> pt.Tensor:
        C = pt.linalg.pinv(self._eigvecs) @ (self._B @ self._cm).type(
            self._eigvals.dtype
        )
        V = pt.linalg.vander(self._eigvals, N=self._n_times - 1)
        dyn = pt.vstack(
            [
                (C[:, :n] * V[:, :n].flip(-1)).sum(dim=1)
                for n in range(1, self._n_times)
            ]
        ).T
        return pt.cat((pt.zeros(self._n_states).unsqueeze(-1), dyn), dim=1)

    @property
    def dynamics(self) -> pt.Tensor:
        return self.unforced_dynamics + self.forced_dynamics

    @property
    def reconstruction(self) -> pt.Tensor:
        return (self._eigvecs @ self.dynamics).type(self._dm.dtype)

    @property
    def reconstruction_error(self) -> pt.Tensor:
        return self._dm - self.reconstruction

    @property
    def projection_error(self) -> pt.Tensor:
        X, Y, G = self._dm[:, :-1], self._dm[:, 1:], self._cm
        return Y - (self._A @ X + self._B @ G)

    @property
    def dft_properties(self) -> Tuple[float, float, float]:
        return _dft_properties(self._dt, self._n_times - 1)
