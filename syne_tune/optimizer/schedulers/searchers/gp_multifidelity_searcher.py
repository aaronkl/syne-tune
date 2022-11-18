# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Optional, List
import logging

from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    gp_multifidelity_searcher_factory,
    gp_multifidelity_searcher_defaults,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import (
    GPFIFOSearcher,
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    ResourceForAcquisitionMap,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    PendingEvaluation,
    MetricValues,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gpiss_model import (
    GaussProcAdditiveModelFactory,
)

logger = logging.getLogger(__name__)


class GPMultiFidelitySearcher(GPFIFOSearcher):
    """
    Gaussian process Bayesian optimization for asynchronous Hyperband scheduler.

    This searcher must be used with :class:`HyperbandScheduler`. It provides a novel
    combination of Bayesian optimization, based on a Gaussian process surrogate
    model, with Hyperband scheduling. In particular, observations across
    resource levels are modelled jointly.

    It is *not* recommended to create :class:`GPMultiFidelitySearcher` searcher
    objects directly, but rather to create :class:`HyperbandScheduler` objects
    with `searcher="bayesopt"`, and passing arguments here in `search_options`.
    This will use the appropriate functions from `gp_searcher_factory.py` to
    create components in a consistent way.

    Most of `GPFIFOSearcher` comments apply here as well.
    In multi-fidelity HPO, we optimize a function f(x, r), x the configuration,
    r the resource (or time) attribute. The latter must be a positive integer.
    In most applications, `resource_attr` == 'epoch', and the resource is the
    number of epochs already trained.

    We model the function f(x, r) jointly over all resource levels r at which
    it is observed (but see `searcher_data` in `HyperbandScheduler`). The kernel
    and mean function of our surrogate model are over (x, r). The surrogate
    model is selected by `gp_resource_kernel`. More details about the supported
    kernels is in:

        Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter and Neural Architecture Search
        https://arxiv.org/abs/2003.10865

    The acquisition function (EI) which is optimized in `get_config`, is obtained
    by fixing the resource level r to a value which is determined depending on
    the current state. If `resource_acq` == 'bohb', r is the largest value
    <= max_t, where we have seen >= dimension(x) metric values. If
    `resource_acq` == 'first', r is the first milestone which config x would
    reach when started.
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        """
        Additional arguments on top of parent class :class:`GPFIFOSearcher`.

        :param model: Selects surrogate model (learning curve model) to be used.
            Choices are:
            * "gp_multitask" (default): GP multi-task surrogate model
            * "gp_independent": Independent GPs for each rung level, sharing
                an ARD kernel
            * "gp_issm": Gaussian-additive model of ISSM type
            * "gp_expdecay": Gaussian-additive model of exponential decay type
                (as in "Freeze Thaw Bayesian Optimization")
        :type model: str, optional
        :param gp_resource_kernel: Only relevant for `model == "gp_multitask"`.
            Surrogate model over criterion function f(x, r), x the config, r the
            resource. Note that x is encoded to be a vector with entries in [0, 1],
            and r is linearly mapped to [0, 1], while the criterion data is
            normalized to mean 0, variance 1. The reference above provides details
            on the models supported here. For the exponential decay kernel, the
            base kernel over x is Matern 5/2 ARD.
            See `SUPPORTED_RESOURCE_MODELS` fur supported choices.
            Defaults to "exp-decay-sum"
        :type gp_resource_kernel: str, optional
        :param resource_acq: Only relevant for `model in {"gp_multitask",
            "gp_independent"}`. Determines how the EI acquisition function is
            used. Values: "bohb", "first". Defaults to "bohb"
        :type resource_acq:
        :param opt_skip_num_max_resource: Parameter for surrogate model fitting,
            skip predicate. If True, and number of observations above
            `opt_skip_init_length`, fitting is done only when there is a new
             datapoint at `r = max_t`, and skipped otherwise. Defaults to False
        :type opt_skip_num_max_resource: bool, optional
        :param issm_gamma_one: Only relevant for `model == "gp_issm"`.
            If True, the gamma parameter of the ISSM is fixed to 1, otherwise it
            is optimized over. Defaults to False
        :type issm_gamma_one: bool, optional
        :param expdecay_normalize_inputs: Only relevant for `model ==
            "gp_expdecay"`. If True, resource values r are normalized to [0, 1]
            as input to the exponential decay surrogate model. Defaults to False
        :type expdecay_normalize_inputs: bool, optional
        """
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_multifidelity_searcher_defaults(), dict_name="search_options"
        )
        kwargs_int = gp_multifidelity_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _call_create_internal(self, kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        k = "resource_for_acquisition"
        self.resource_for_acquisition = kwargs_int.get(k)
        if self.resource_for_acquisition is not None:
            kwargs_int.pop(k)
            assert isinstance(self.resource_for_acquisition, ResourceForAcquisitionMap)
        self.config_space_ext = kwargs_int.pop("config_space_ext")
        self._create_internal(**kwargs_int)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
        from syne_tune.optimizer.schedulers.synchronous.hyperband import (
            SynchronousHyperbandScheduler,
        )

        super().configure_scheduler(scheduler)
        assert isinstance(
            scheduler, (HyperbandScheduler, SynchronousHyperbandScheduler)
        ), (
            "This searcher requires HyperbandScheduler or "
            + "SynchronousHyperbandScheduler scheduler"
        )
        self._resource_attr = scheduler.resource_attr
        model_factory = self.state_transformer.model_factory
        if isinstance(model_factory, GaussProcAdditiveModelFactory):
            assert scheduler.searcher_data == "all", (
                "For an additive Gaussian learning curve model (model="
                + "'gp_issm' or model='gp_expdecay' in search_options), you "
                + "need to set searcher_data='all' when creating the "
                + "HyperbandScheduler"
            )

    def _hp_ranges_in_state(self):
        return self.config_space_ext.hp_ranges_ext

    def _config_ext_update(self, config, result):
        resource = int(result[self._resource_attr])
        return self.config_space_ext.get(config, resource)

    def _metric_val_update(self, crit_val: float, result: dict) -> MetricValues:
        resource = result[self._resource_attr]
        return {str(resource): crit_val}

    def _trial_id_string(self, trial_id: str, result: dict):
        """
        For multi-fidelity, we also want to output the resource level
        """
        return f"{trial_id}:{result[self._resource_attr]}"

    def register_pending(
        self,
        trial_id: str,
        config: Optional[dict] = None,
        milestone: Optional[int] = None,
    ):
        assert (
            milestone is not None
        ), "This searcher works with a multi-fidelity scheduler only"
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        if not state.is_pending(trial_id, resource=milestone):
            assert not state.is_labeled(trial_id, resource=milestone), (
                f"Trial trial_id = {trial_id} already has observation at "
                + f"resource = {milestone}, so cannot be pending there"
            )
            self.state_transformer.append_trial(
                trial_id, config=config, resource=milestone
            )

    def _fix_resource_attribute(self, **kwargs):
        """
        Determines target resource level r at which the current call of
        `get_config` operates. This is done based on
        `resource_for_acquisition`. This resource level is then set in
        `config_space_ext.hp_ranges_ext.value_for_last_pos`. This does the
        job for GP surrogate models. But if in subclasses, other surrogate
        models are involved, they need to get informed separately (see
        :class:`CostAwareGPMultiFidelitySearcher` for an example).
        """
        if self.resource_for_acquisition is not None:
            # Only have to do this for 'gp_multitask' or 'gp_independent' model
            state = self.state_transformer.state
            # BO should only search over configs at resource level
            # target_resource
            if state.trials_evaluations:
                target_resource = self.resource_for_acquisition(state, **kwargs)
            else:
                # Any valid value works here:
                target_resource = self.config_space_ext.resource_attr_range[0]
            self.config_space_ext.hp_ranges_ext.value_for_last_pos = target_resource
            if self.debug_log is not None:
                self.debug_log.append_extra(
                    f"Score values computed at target_resource = {target_resource}"
                )

    def _postprocess_config(self, config: dict) -> dict:
        # If `config` is normal (not extended), nothing is removed
        return self.config_space_ext.remove_resource(config)

    def evaluation_failed(self, trial_id: str):
        # Remove all pending evaluations for trial
        self.cleanup_pending(trial_id)
        # Mark config as failed (which means it will not be suggested again)
        self.state_transformer.mark_trial_failed(trial_id)

    def cleanup_pending(self, trial_id: str):
        def filter_pred(x: PendingEvaluation) -> bool:
            return x.trial_id == trial_id

        self.state_transformer.filter_pending_evaluations(filter_pred)

    def remove_case(self, trial_id: str, **kwargs):
        resource = kwargs[self._resource_attr]
        self.state_transformer.remove_observed_case(trial_id, key=str(resource))

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        skip_optimization = state["skip_optimization"]
        model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = GPMultiFidelitySearcher(
            **self._new_searcher_kwargs_for_clone(),
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
            config_space_ext=self.config_space_ext,
            resource_for_acquisition=self.resource_for_acquisition,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
