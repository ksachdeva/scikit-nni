#!/usr/bin/env python

import unittest
import yaml

from sknni.internals import PipelineBuilder
from sklearn.preprocessing import Normalizer

from lightgbm import LGBMClassifier


class TestPipelineBuilder(unittest.TestCase):
    """Tests for `sknni` package."""
    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_estimator_arguments(self):
        """ Tests setting the arguments which are not in the search space """
        simple_config = yaml.load("""

            sklearnPipeline:
                name: normalizer_lightgbm
                steps:
                    normalizer:
                        type: sklearn.preprocessing.Normalizer
                    lightgbm:
                        type: lightgbm.LGBMClassifier
                        classArgs:
                            objective: multiclass

            nniConfigSearchSpace:
                - lightgbm:
                    num_leaves:
                        _type: choice
                        _value: [31,41,51]
                    boosting_type:
                        _type: choice
                        _value: [gbdt, goss, dart]

        """)

        print(simple_config)

        pipeline = PipelineBuilder(simple_config)({
            'lightgbm_num_leaves':
            31,
            'lightgbm_boosting_type':
            'goss'
        })

        assert len(pipeline.named_steps.keys()) == 2
        assert isinstance(pipeline.named_steps['normalizer'], Normalizer)
        assert isinstance(pipeline.named_steps['lightgbm'], LGBMClassifier)

        assert pipeline.named_steps['lightgbm'].num_leaves == 31
        assert pipeline.named_steps['lightgbm'].objective == 'multiclass'
        assert pipeline.named_steps['lightgbm'].boosting_type == 'goss'

    def test_command_line_interface(self):
        pass
