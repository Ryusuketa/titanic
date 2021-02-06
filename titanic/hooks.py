from typing import Any, Dict, Iterable, Optional

from kedro.config import TemplatedConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from titanic.pipelines import data_engineering as de
from titanic.pipelines import data_handlers as dh
from titanic.pipelines import data_science as ds


class ProjectHooks:
    _mode: str = ''

    @classmethod
    def set_mode(cls, mode: str):
        cls._mode = mode

    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        data_wrangler_pipeline = de.create_data_wrangler_pipeline()
        data_merge_pipeline = de.create_data_merge_pipeline()
        data_handler_pipeline = dh.create_data_handler()
        training_pipeline = ds.create_training_pipeline()
        evaluation_pipeline = ds.create_evaluation_pipeline()

        inference_data_wrangler_pipeline = de.create_data_wrangler_pipeline('inference')
        inference_data_merge_pipeline = de.create_data_merge_pipeline('inference')
        inference_data_loader_pipeline = dh.create_inference_data_handler()
        inference_pipeline = ds.create_inference_pipeline()
        return {
            'de': data_wrangler_pipeline,
            'dm': data_merge_pipeline,
            'dh': data_handler_pipeline,
            'train': training_pipeline,
            'evaluation': evaluation_pipeline,
            'inference': (inference_data_wrangler_pipeline + inference_data_merge_pipeline +
                          inference_data_loader_pipeline + inference_pipeline),
            '__default__': (data_wrangler_pipeline + data_merge_pipeline + data_handler_pipeline + training_pipeline +
                            evaluation_pipeline)
        }

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> TemplatedConfigLoader:
        return TemplatedConfigLoader(conf_paths, globals_dict=dict(mode=self._mode))

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


project_hooks = ProjectHooks()
