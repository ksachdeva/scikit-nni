""" Pipeline Builder """

import yaml

from sklearn.pipeline import Pipeline

from ._utils import get_class

class PipelineBuilder(object):
    def __init__(self, experiment_spec):
        self.steps = experiment_spec['sklearnPipeline']['steps']
        self.params_info = self._param_info_from_search_space(experiment_spec['nniConfigSearchSpace'])

    def _param_info_from_search_space(self, search_space):
        steps_with_params = {}
        for el in search_space:
            for k, v in el.items():
                if v is None:
                    continue
                # value is the dictionary with key being the names
                # of the params
                steps_with_params[k] = list(v.keys())

        return steps_with_params

    def __call__(self, nni_hparams):
        # need to first create and feed the corresponding hparams to
        # a step in the pipeline
        sklearn_steps = []
        for k, v in self.steps.items():
            estimator_cls = get_class(v)

            # find the arguments for this estimator and set their values
            # using nni_hparams
            if not k in self.params_info.keys():
                # does not have any parameter
                sklearn_steps.append((k, estimator_cls()))
                continue

            kwargs = {}
            for p in self.params_info[k]:
                kwargs[p] = nni_hparams[f"{k}_{p}"]

            sklearn_steps.append((k, estimator_cls(**kwargs)))

        return Pipeline(steps=sklearn_steps)
