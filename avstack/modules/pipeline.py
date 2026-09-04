from typing import Any, Dict, List

from avstack.config import MODELS, PIPELINE, ConfigDict
from avstack.utils.decorators import apply_hooks

from .base import BaseModule


@PIPELINE.register_module()
class SerialPipeline(BaseModule):
    def __init__(self, modules: List[ConfigDict], *args: Any, **kwargs: Any) -> None:
        super().__init__(name="pipeline", *args, **kwargs)
        self.modules = [
            MODELS.build(mod) if isinstance(mod, dict) else mod for mod in modules
        ]

    @apply_hooks
    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        for module in self.modules:
            data = module(data, *args, **kwargs)
        return data

    def initialize(self, *args, **kwargs):
        for module in self.modules:
            module.initialize(*args, **kwargs)


@PIPELINE.register_module()
class ModularDrivingPipeline(BaseModule):
    """Standard modular AV driving stack: perception -> tracking -> planning -> control.

    Maps raw sensor data plus the ego vehicle state to a vehicle control command by running four
    swappable avstack stages in series. This is the modular counterpart to end-to-end or
    foundation-model driving stacks: each stage is an independent module built from config, so a
    security test can attach an attack/defense as a pre/post hook on any stage (e.g. a spoofed
    detection on ``perception``) and watch it propagate through tracking and planning into control.

    Called as ``pipeline(sensor_data, ego_state)``. The closed-loop bridge (e.g. avcarla's mobile
    actor) supplies the freshest sensor bundle and ego state each tick; when the bridge hands over a
    ``{sensor_id: data}`` bundle, ``perception_input`` selects the sensor feeding perception (a lone
    sensor is used automatically).
    """

    def __init__(
        self,
        perception: ConfigDict,
        tracking: ConfigDict,
        planning: ConfigDict,
        control: ConfigDict,
        perception_input: str = None,
        waypoint: Dict[str, Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="modular_driving", *args, **kwargs)
        from .planning.types import WaypointPlan

        self.perception = MODELS.build(perception)
        self.tracking = MODELS.build(tracking)
        self.planning = MODELS.build(planning)
        self.control = MODELS.build(control)
        self.perception_input = perception_input
        self.plan = WaypointPlan(**(waypoint or {}))
        self._last_data = None

    def _perception_data(self, data: Any) -> Any:
        if isinstance(data, dict):
            if self.perception_input is not None:
                data = data.get(self.perception_input)
            else:
                present = [v for v in data.values() if v is not None]
                if len(present) > 1:
                    raise ValueError(
                        "ModularDrivingPipeline received a multi-sensor bundle "
                        f"({list(data)}); set `perception_input` to choose one."
                    )
                data = present[0] if present else None
        # a closed-loop sensor can drop a frame (async delivery); coast on the last good cloud
        if data is None:
            data = self._last_data
        else:
            self._last_data = data
        return data

    @apply_hooks
    def __call__(self, data: Any, ego_state: Any, *args: Any, **kwargs: Any) -> Any:
        detections = self.perception(self._perception_data(data))
        self.tracking(detections, platform=ego_state.reference)
        objects = self.tracking.tracks_confirmed
        self.planning(self.plan, ego_state, objects)
        return self.control(ego_state, self.plan)

    def initialize(self, t0=None, ego_state=None, destination=None, map_data=None, *a, **k):
        # the straight-drive stack needs no per-module warmup; keep the route context for planners
        self.destination = destination
        self.map_data = map_data


@PIPELINE.register_module()
class MappedPipeline(BaseModule):
    def __init__(
        self,
        modules: Dict[str, ConfigDict],
        mapping: Dict[str, List[str]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A non-serial pipeline of modules mapping data to MODELS

        Can support routing data between multiple modules
        Gets tricky if the ordering of inputs to modules is important

        Arguments:
        :modules - dictionary of names to MODELS
        :mapping - dictionary of names of MODELS to names of MODELS whose outputs form inputs

        Example:
        ---
        modules = {"percep1": ALG1, "percep2": ALG2, "tracking": ALG3}
        mapping = {"percep1": ["sensor1"], "percep2": ["sensor2"], "tracking": ["percep1", "percep2"]}
        pipeline = MappedPipeline(modules, pipeline)
        data_in = {"sensor1": DATA1, "sensor2": DATA2}
        tracks = pipeline(data_in)
        """
        super().__init__(name="pipeline", *args, **kwargs)
        self.modules = {
            name: MODELS.build(mod) if isinstance(mod, dict) else mod
            for name, mod in modules.items()
        }
        self.mapping = mapping

    @apply_hooks
    def __call__(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Runs modules one-by-one in order mapping data between them"""
        for name, module in self.modules.items():
            this_in = [data[in_name] for in_name in self.mapping[name]]
            last_data = module(*this_in, *args, **kwargs)
            data[name] = last_data
        return last_data  # only return last module data?

    def initialize(self, *args, **kwargs):
        for module in self.modules.items():
            module.initialize(*args, **kwargs)


@PIPELINE.register_module()
class CustomPipeline(BaseModule):
    def __init__(self, modules: List[ConfigDict], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.modules = [MODELS.build(mod) for mod in modules]

    @apply_hooks
    def __call__(self):
        raise NotImplementedError("Implement a custom pipeline in a subclass")

    def initialize(self, *args, **kwargs):
        for module in self.modules:
            module.initialize(*args, **kwargs)
