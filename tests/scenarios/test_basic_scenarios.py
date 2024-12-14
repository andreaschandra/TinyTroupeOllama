import logging

import pytest

logger = logging.getLogger("tinytroupe")

import sys

sys.path.append("../../tinytroupe/")
sys.path.append("../../")
sys.path.append("..")


import tinytroupe
import tinytroupe.control as control
from testing_utils import *
from tinytroupe.agent import TinyPerson, TinyToolUse
from tinytroupe.control import Simulation
from tinytroupe.enrichment import TinyEnricher
from tinytroupe.environment import TinySocialNetwork, TinyWorld
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_marcos_the_physician,
    create_oscar_the_architect,
)
from tinytroupe.extraction import ArtifactExporter, ResultsExtractor
from tinytroupe.extraction import default_extractor as extractor
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.tools import TinyWordProcessor


def test_basic_scenario_1():
    control.reset()

    assert (
        control._current_simulations["default"] is None
    ), "There should be no simulation running at this point."

    control.begin()
    assert (
        control._current_simulations["default"].status == Simulation.STATUS_STARTED
    ), "The simulation should be started at this point."

    agent = create_oscar_the_architect()

    agent.define("age", 19)
    agent.define("nationality", "Brazilian")

    assert (
        control._current_simulations["default"].cached_trace is not None
    ), "There should be a cached trace at this point."
    assert (
        control._current_simulations["default"].execution_trace is not None
    ), "There should be an execution trace at this point."

    control.checkpoint()
    # TODO check file creation

    agent.listen_and_act("How are you doing?")
    agent.define("occupation", "Engineer")

    control.checkpoint()
    # TODO check file creation

    control.end()


def test_tool_usage_1():

    data_export_folder = f"{EXPORT_BASE_FOLDER}/test_tool_usage_1"

    exporter = ArtifactExporter(base_output_folder=data_export_folder)
    enricher = TinyEnricher()
    tooluse_faculty = TinyToolUse(
        tools=[TinyWordProcessor(exporter=exporter, enricher=enricher)]
    )

    lisa = create_lisa_the_data_scientist()

    lisa.add_mental_faculties([tooluse_faculty])

    actions = lisa.listen_and_act(
        """
                            You have just been fired and need to find a new job. You decide to think about what you 
                            want in life and then write a resume. The file must be titled 'Resume'.
                            Don't stop until you actually write the resume.
                            """,
        return_actions=True,
    )

    assert contains_action_type(
        actions, "WRITE_DOCUMENT"
    ), "There should be a WRITE_DOCUMENT action in the actions list."

    # check that the document was written to a file
    assert os.path.exists(
        f"{data_export_folder}/Document/Resume.docx"
    ), "The document should have been written to a file."
    assert os.path.exists(
        f"{data_export_folder}/Document/Resume.json"
    ), "The document should have been written to a file."
    assert os.path.exists(
        f"{data_export_folder}/Document/Resume.md"
    ), "The document should have been written to a file."

    assert (
        control._current_simulations["default"].cached_trace is not None
    ), "There should be a cached trace at this point."
    assert (
        control._current_simulations["default"].execution_trace is not None
    ), "There should be an execution trace at this point."
