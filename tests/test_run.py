from pathlib import Path

import pytest

from titanic.run import ProjectContext


@pytest.fixture
def project_context(mocker):
    # Don't configure the logging module. If it's configured, tests that
    # check logs using the ``caplog`` fixture depend on execution order.
    mocker.patch.object(ProjectContext, "_setup_logging")

    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "titanic"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.16.6"
