"""Provide classes and functions for modeling and evaluating polarised emission.

It includes:
- FittableEmissionModel: A base class for two-dimensional fittable emission models.
- FittablePolarisationModel: A base class for two-dimensional fittable polarisation models.
- fittablePolarisedEmissionModel: A model for fittable polarised emission.
- compoundFittablePolarisedEmissionModel: A compound model for polarised emission.

It includes:
- FittableEmissionModel: A base class for two-dimensional fittable emission models.
- FittablePolarisationModel: A base class for two-dimensional fittable polarisation models.
- fittablePolarisedEmissionModel: A model for fittable polarised emission.
- compoundFittablePolarisedEmissionModel: A compound model for polarised emission.
"""

import numpy as np
from astropy.modeling import CompoundModel, FittableModel

from mapext.core.stokes import StokesComp, wrap_theta

__all__ = [
    "FittableEmissionModel",
    "FittablePolarisationModel",
    "compoundFittablePolarisedEmissionModel",
    "fittablePolarisedEmissionModel",
]


class FittableEmissionModel(FittableModel):
    """Base class for two-dimensional fittable models.

    This class provides an easier interface for defining new models.
    """

    n_inputs = 2
    n_outputs = 1

    def deriv(self, nu, beam):
        """Compute the derivative of the model.

        Parameters
        ----------
        nu : array-like
            Frequency values.
        beam : array-like
            Beam values.

        Returns
        -------
        array-like
            The computed derivative of the model.
        """
        args = [getattr(self, p).value for p in self.param_names]
        return self.fit_deriv(nu, beam, *args)


class FittablePolarisationModel(FittableModel):
    """Base class for two-dimensional fittable models.

    This class provides an easier interface for defining new models.
    """

    n_inputs = 3  # nu, beam, stokes
    n_outputs = 1

    def deriv(self, nu, beam, stokes):
        """Compute the derivative of the model.

        Parameters
        ----------
        nu : array-like
            Frequency values.
        beam : array-like
            Beam values.
        stokes : array-like
            Stokes parameter values.

        Returns
        -------
        array-like
            The computed derivative of the model.
        """
        args = [getattr(self, p).value for p in self.param_names]
        return self.fit_deriv(nu, beam, stokes, *args)


class fittablePolarisedEmissionModel(CompoundModel):
    """A model for fittable polarised emission.

    This class combines an emission model and a polarisation model to evaluate
    polarised emission based on Stokes parameters. It supports operations like
    multiplication and provides methods for evaluating the combined model.
    """

    def __init__(self, emission_model, polarisation_model, name=None):
        self.__dict__["_param_names"] = None
        self._n_submodels = None
        self.left = emission_model
        self.right = polarisation_model
        self.op = "*"  # set to be multiply
        self._bounding_box = None
        self._user_bounding_box = None
        self._leaflist = None
        self._tdict = None
        self._parameters = None
        self._parameters_ = None
        self._param_metrics = None

        self._n_models = len(polarisation_model)

        if (
            emission_model.model_set_axis != polarisation_model.model_set_axis
        ) or emission_model.model_set_axis:  # not False and not 0
            raise ValueError(
                "model_set_axis must be False or 0 and consistent for operands"
            )
        self._model_set_axis = emission_model.model_set_axis

        self.n_inputs = polarisation_model.n_inputs
        self.n_outputs = polarisation_model.n_outputs
        self.inputs = polarisation_model.inputs
        self.outputs = polarisation_model.outputs

        self.name = name
        self._fittable = None
        self.linear = False
        self.eqcons = []
        self.ineqcons = []
        self.n_left_params = len(self.left.parameters)
        self._map_parameters()

        # Initialize the cache for the constraints (used primarily when
        # sync_constraints is False)
        self._constraints_cache = {}

    def _evaluate(self, *args, **kw):
        args, kw = self._get_kwarg_model_parameters_as_positional(args, kw)
        # sort emission model
        left_inputs = self._get_left_inputs_from_args(args)
        left_params = self._get_left_params_from_args(args)
        leftval = np.array(self.left.evaluate(*left_inputs, *left_params))

        # sort polarisation model
        right_inputs = self._get_right_inputs_from_args(args)
        right_params = self._get_right_params_from_args(args)
        rightval = np.array(self.right.evaluate(*right_inputs, *right_params))

        stokes = right_inputs[-1]
        outarray = np.full(right_inputs[0].shape, np.nan)

        outarray[stokes == int(StokesComp("I"))] = leftval[
            stokes == int(StokesComp("I"))
        ]
        outarray[stokes == int(StokesComp("Q"))] = (
            leftval[stokes == int(StokesComp("Q"))]
            * rightval[stokes == int(StokesComp("Q"))]
        )
        outarray[stokes == int(StokesComp("U"))] = (
            leftval[stokes == int(StokesComp("U"))]
            * rightval[stokes == int(StokesComp("U"))]
        )
        outarray[stokes == int(StokesComp("V"))] = (
            leftval[stokes == int(StokesComp("V"))]
            * rightval[stokes == int(StokesComp("V"))]
        )
        outarray[stokes == int(StokesComp("P"))] = (
            leftval[stokes == int(StokesComp("P"))]
            * rightval[stokes == int(StokesComp("P"))]
        )
        outarray[stokes == int(StokesComp("A"))] = wrap_theta(
            rightval[stokes == int(StokesComp("A"))]
        )
        outarray[stokes == int(StokesComp("PF"))] = rightval[
            stokes == int(StokesComp("PF"))
        ]

        return outarray

    def _pre_evaluate(self, *args, **kwargs):
        """Model specific input setup that needs to occur prior to model evaluation."""
        # Broadcast inputs into common size
        inputs, broadcasted_shapes = self.prepare_inputs(*args, **kwargs)

        # Setup actual model evaluation method
        parameters = self._param_sets(raw=True, units=self._has_units)

        def evaluate(_inputs):
            return self._evaluate(*_inputs, *parameters)

        return evaluate, inputs, broadcasted_shapes, kwargs

    def _get_kwarg_model_parameters_as_positional(self, args, kwargs):
        new_args = list(args[: self.n_inputs])
        args_pos = self.n_inputs

        for param_name in self.param_names:
            kw_value = kwargs.pop(param_name, None)
            if kw_value is not None:
                value = kw_value
            else:
                try:
                    value = args[args_pos]
                except IndexError:
                    raise IndexError("Missing parameter or input")

                args_pos += 1
            new_args.append(value)

        return new_args, kwargs

    def _get_left_inputs_from_args(self, args):
        out = args[: self.left.n_inputs]
        return np.broadcast_arrays(*out)

    def _get_left_params_from_args(self, args):
        return args[self.n_inputs : self.n_inputs + self.n_left_params]

    def _get_right_inputs_from_args(self, args):
        out = args[: self.right.n_inputs]
        return np.broadcast_arrays(*out)

    def _get_right_params_from_args(self, args):
        return args[self.n_inputs + self.n_left_params :]


def _FPEM_oper(oper, **kwargs):
    """Returns a function that evaluates a given Python arithmetic operator.

    The operator should be given as a string, like ``'+'`` or ``'**'``.
    """
    return lambda left, right: compoundFittablePolarisedEmissionModel(
        oper, left, right, **kwargs
    )


setattr(fittablePolarisedEmissionModel, "__add__", _FPEM_oper("+"))
delattr(fittablePolarisedEmissionModel, "__sub__")
delattr(fittablePolarisedEmissionModel, "__mul__")
delattr(fittablePolarisedEmissionModel, "__truediv__")
delattr(fittablePolarisedEmissionModel, "__pow__")
delattr(fittablePolarisedEmissionModel, "__or__")
delattr(fittablePolarisedEmissionModel, "__and__")


class compoundFittablePolarisedEmissionModel(CompoundModel):
    """A compound model for fittable polarised emission.

    This class combines two models (left and right) to evaluate polarised emission
    based on Stokes parameters. It supports operations like addition and provides
    methods for evaluating the combined model.
    """

    def _evaluate(self, *args, **kw):
        freq, beam, stokes = args
        try:
            stokes = int(stokes)
        except ValueError:
            stokes = stokes.astype(int)
        freq, beam, stokes = np.broadcast_arrays(
            np.array(freq), np.array(beam), np.array(stokes)
        )

        left_inputs = np.array([freq, beam, stokes])
        right_inputs = np.array([freq, beam, stokes])
        left_params = kw
        right_params = kw

        leftval = self.left(*left_inputs, **kw)
        rightval = self.right(*right_inputs, **kw)

        outarray = np.full(stokes.shape, np.nan)

        outarray[stokes == int(StokesComp("I"))] = (
            leftval[stokes == int(StokesComp("I"))]
            + rightval[stokes == int(StokesComp("I"))]
        )
        outarray[stokes == int(StokesComp("Q"))] = (
            leftval[stokes == int(StokesComp("Q"))]
            + rightval[stokes == int(StokesComp("Q"))]
        )
        outarray[stokes == int(StokesComp("U"))] = (
            leftval[stokes == int(StokesComp("U"))]
            + rightval[stokes == int(StokesComp("U"))]
        )
        outarray[stokes == int(StokesComp("V"))] = (
            leftval[stokes == int(StokesComp("V"))]
            + rightval[stokes == int(StokesComp("V"))]
        )

        # P must be calculated
        if np.any(stokes == int(StokesComp("P"))):
            left_ins = left_inputs[:, stokes == int(StokesComp("P"))]
            right_ins = right_inputs[:, stokes == int(StokesComp("P"))]

            left_q = np.array(self.left(*left_ins[:-1], StokesComp("Q"), *left_params))
            left_u = np.array(self.left(*left_ins[:-1], StokesComp("U"), *left_params))
            left_v = np.array(self.left(*left_ins[:-1], StokesComp("V"), *left_params))

            right_q = np.array(
                self.right(*right_ins[:-1], StokesComp("Q"), *right_params)
            )
            right_u = np.array(
                self.right(*right_ins[:-1], StokesComp("U"), *right_params)
            )
            right_v = np.array(
                self.right(*right_ins[:-1], StokesComp("V"), *right_params)
            )

            outarray[stokes == int(StokesComp("P"))] = np.sqrt(
                (left_q + right_q) ** 2
                + (left_u + right_u) ** 2
                + (left_v + right_v) ** 2
            )

        # A must be calculated
        if np.any(stokes == int(StokesComp("A"))):
            left_ins = left_inputs[:, stokes == int(StokesComp("A"))]
            right_ins = right_inputs[:, stokes == int(StokesComp("A"))]

            left_q = np.array(self.left(*left_ins[:-1], StokesComp("Q"), *left_params))
            left_u = np.array(self.left(*left_ins[:-1], StokesComp("U"), *left_params))

            right_q = np.array(
                self.right(*right_ins[:-1], StokesComp("Q"), *right_params)
            )
            right_u = np.array(
                self.right(*right_ins[:-1], StokesComp("U"), *right_params)
            )

            a = 0.5 * np.arctan2(left_u + right_u, left_q + right_q)
            outarray[stokes == int(StokesComp("A"))] = wrap_theta(a)

        # PF must be calculated
        if np.any(stokes == int(StokesComp("PF"))):
            left_ins = left_inputs[:, stokes == int(StokesComp("A"))]
            right_ins = right_inputs[:, stokes == int(StokesComp("A"))]

            left_i = np.array(self.left(*left_ins[:-1], StokesComp("I"), *left_params))
            left_q = np.array(self.left(*left_ins[:-1], StokesComp("Q"), *left_params))
            left_u = np.array(self.left(*left_ins[:-1], StokesComp("U"), *left_params))
            left_v = np.array(self.left(*left_ins[:-1], StokesComp("V"), *left_params))

            right_i = np.array(
                self.right(*right_ins[:-1], StokesComp("I"), *right_params)
            )
            right_q = np.array(
                self.right(*right_ins[:-1], StokesComp("Q"), *right_params)
            )
            right_u = np.array(
                self.right(*right_ins[:-1], StokesComp("U"), *right_params)
            )
            right_v = np.array(
                self.right(*right_ins[:-1], StokesComp("V"), *right_params)
            )

            total_i = left_i + right_i
            total_p = np.sqrt(
                (left_q + right_q) ** 2
                + (left_u + right_u) ** 2
                + (left_v + right_v) ** 2
            )
            outarray[stokes == int(StokesComp("PF"))] = total_p / total_i

        return outarray


setattr(compoundFittablePolarisedEmissionModel, "__add__", _FPEM_oper("+"))
delattr(compoundFittablePolarisedEmissionModel, "__sub__")
delattr(compoundFittablePolarisedEmissionModel, "__mul__")
delattr(compoundFittablePolarisedEmissionModel, "__truediv__")
delattr(compoundFittablePolarisedEmissionModel, "__pow__")
delattr(compoundFittablePolarisedEmissionModel, "__or__")
delattr(compoundFittablePolarisedEmissionModel, "__and__")
