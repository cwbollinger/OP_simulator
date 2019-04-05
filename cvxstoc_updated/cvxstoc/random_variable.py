import pymc3 as pm

import numpy
import scipy

from . import strings
import cvxpy.expressions.constants.constant
import cvxpy.lin_ops.lin_utils


class RandomVariableFactory:
    _id = 0

    def __init__(self):
        pass

    # Note: alpha here is in R_+^K <==> x is in R^K ~ Dir(alpha)
    def create_dirichlet_rv(self, alpha):
        rv_name = self.get_rv_name()

        rv_pymc = pm.Dirichlet(name=rv_name, theta=alpha)

        metadata = {}
        metadata['mu'] = [alpha[i] / numpy.sum(alpha) for i in range(len(alpha))]

        metadata['alpha'] = alpha

        return RandomVariable(rv=rv_pymc, metadata=metadata)

    def create_unif_rv(self, lower=0, upper=1, cont=True, shape=1):
        rv_name = self.get_rv_name()

        rv_pymc = None
        if cont:
            rv_pymc = pm.Uniform(name=rv_name, lower=lower, upper=upper, size=shape)
        else:
            rv_pymc = pm.DiscreteUniform(name=rv_name, lower=lower, upper=upper, size=shape)

        metadata = {}
        metadata['mu'] = (1.0 / 2.0) * (upper + lower)

        metadata['lower'] = lower
        metadata['upper'] = upper

        return RandomVariable(rv=rv_pymc, metadata=metadata)

    def create_categorical_rv(self, vals=None, probs=None, shape=1):
        rv_name = self.get_rv_name()

        rv_pymc = pm.Categorical(name=rv_name, p=list(probs), size=shape)

        metadata = {}
        metadata["mu"] = len(probs) * numpy.asarray(probs)

        val_map = {}
        # True when vals != None && vals != []
        if vals:
            for val in range(0, len(probs)):
                val_map[val] = vals[val]
        metadata["vals"] = vals
        metadata["probs"] = probs

        return RandomVariable(rv=rv_pymc, val_map=val_map, metadata=metadata)

    def create_normal_rv(self, mu, cov, shape=None):
        # print('hello :)')
        rv_name = self.get_rv_name()

        metadata = {}
        # print('there :)')

        model = pm.Model()
        with model:

            pm.MvNormal(rv_name, mu=mu, cov=cov, shape=mu.shape)
            # print('general :)')

        metadata['mu'] = mu
        metadata['cov'] = cov
        # print('kenobi :)')

        return RandomVariable(model=model, metadata=metadata, name=rv_name)

    def get_rv_name(self):
        return "rv" + str(RandomVariableFactory.get_next_avail_id())

    @staticmethod
    def get_next_avail_id():
        RandomVariableFactory._id += 1
        return RandomVariableFactory._id


def NormalRandomVariable(mean, cov, shape=1):
    return RandomVariableFactory().create_normal_rv(mu=mean, cov=cov, shape=shape)


def CategoricalRandomVariable(vals, probs, shape=1):
    return RandomVariableFactory().create_categorical_rv(vals=vals, probs=probs, shape=shape)


def UniformRandomVariable(lower=0, upper=1, cont=True, shape=1):
    return RandomVariableFactory().create_unif_rv(lower=lower, upper=upper, cont=cont, shape=shape)


class RandomVariable(cvxpy.expressions.constants.parameter.Parameter):

    # model == pymc.Model object
    def __init__(self, rv=None, model=None, name=None, val_map=None, metadata=None):
        if name is not None:
            self._name = name

        self._metadata = metadata

        self._val_map = val_map

        self.set_rv_model_and_maybe_name(rv, model)

        shape = self.get_shape()

        # super(RandomVariable, self).__init__(shape=shape, name=self._name, value=self.mean)
        super(RandomVariable, self).__init__(shape=shape, name=self._name)
        # print('Shape: {}'.format(self.shape))
        # print('Shape mu: {}'.format(self.mean.shape))

    @property
    def mean(self):
        return self._metadata['mu']

    def set_rv_model_and_maybe_name(self, rv, model):
        # first case seems like the only one used
        if rv is not None and model is None:
            self._rv = rv
            self._model = pm.Model([self._rv])

            if self._name is None:
                self._name = self._rv.name

        elif rv is not None and model is not None:
            self._rv = rv
            self._model = model

            self._name = self._rv.name

        elif rv is None and model is not None:

            self._model = model

            self._rv = None
            # print(dir(model))
            # print(model.unobserved_RVs)

            variables = self._model.unobserved_RVs + self._model.observed_RVs
            for pymc_variable in variables:
                if pymc_variable.name == self._name:
                    # going to try doing none of this other stuff...
                    # if isinstance(pymc_variable, pm.CompletedDirichlet):
                    #     if pymc_variable.parents["D"].observed:
                    #         continue
                    #     # Success.
                    # elif hasattr(pymc_variable, "observed"):
                    #     if pymc_variable.observed:
                    #         continue
                    #     # Success.
                    # # Failure.
                    # else:
                    #     raise Exception(strings.UNSUPPORTED_PYMC_RV)
                    self._rv = pymc_variable
                    break

            if self._rv is None:
                raise Exception(strings.CANT_FIND_PYMC_RV_IN_PYMC_MODEL_OBJ)

        else:
            raise Exception(strings.DIDNT_PASS_EITHER_RV_OR_MODEL)

    def get_shape(self):
        shape = ()
        if self.has_val_map():
            val = self._val_map.values()[0]

            if isinstance(val, int) or isinstance(val, float):
                shape = (1, 1)

            elif isinstance(val, numpy.ndarray):

                shape = numpy_shape = val.shape
                if len(numpy_shape) == 1:
                    shape = (numpy_shape[0], 1)
                elif len(numpy_shape) == 2:
                    shape = (numpy_shape[0], numpy_shape[1])
                else:
                    raise Exception(strings.BAD_RV_DIMS)

            else:
                raise Exception(strings.BAD_VAL_MAP)

        else:
            pymc_shape = ()
            if isinstance(self._rv, pm.Dirichlet):
                pymc_shape = self._rv.parents["D"].shape
            else:
                pymc_shape = self._rv.tag.test_value.shape

            if len(pymc_shape) == 0:
                shape = (1,)
            elif len(pymc_shape) == 1:
                shape = pymc_shape
            elif len(pymc_shape) == 2:
                shape = pymc_shape
            else:
                raise Exception(strings.BAD_RV_DIMS)

        return shape

    def name(self):
        # Override.
        if self.value is None:
            return self._name
        else:
            return str(self.value)

    def __repr__(self):
        # Override.
        return "RandomVariable(%s, %s, %s)" % (self.curvature, self.sign, self.size)

    def __eq__(self, rv):
        # Override.
        return self._name == rv._name

    def __hash__(self):
        # Override.
        return hash(self._name)

    def __deepcopy__(self, memo):
        # Override.
        return self.__class__(rv=self._rv,
                              model=self._model,
                              name=self.name(),
                              val_map=self._val_map,
                              metadata=self._metadata)

    def sample(self, num_samples, num_burnin_samples=0):
        if num_samples == 0:
            return [None]

        with self._model as model:
            # mcmc = pm.MCMC(self._model)
            # samples = pm.sample(num_samples, num_burnin_samples, progress_bar=False)
            samples = pm.sample(num_samples)
            # samples = mcmc.trace(self._name)[:]

        if not self.has_val_map():
            return samples
        else:
            samples_mapped = [self._val_map[sample[0]] for sample in samples]
            return samples_mapped

    def has_val_map(self):
        if self._val_map is not None and len(self._val_map.values()) > 0:
            return True
        else:
            return False
