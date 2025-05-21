import sys
import os
import tempfile
import json


def test_setup_terminal_colors():
    from cvrmap.utils.tools import setup_terminal_colors
    assert setup_terminal_colors() == None


def test_parse_args():
    from cvrmap.utils.tools import parse_args
    sys.argv = ["cmd",
                "bids_dir",
                "output_dir",
                "participant",
                "--derivatives",
                "fmriprep=/path/to/fmriprep"]
    args = parse_args()
    assert args.derivatives[0] == "fmriprep=/path/to/fmriprep"


def test_is_fmriprep_old_version():
    from cvrmap.utils.tools import is_fmriprep_old_version

    with tempfile.TemporaryDirectory() as temp_dir:

        fmriprep_temp_dir = os.path.join(temp_dir, "fmriprep")

        os.mkdir(fmriprep_temp_dir)

        dataset_description = {"GeneratedBy": {"Name": "fMRIPrep",
                                               "Version": "21.0.4"},}

        with open(os.path.join(fmriprep_temp_dir, "dataset_description.json"), "w") as json_file:
            json.dump(dataset_description, json_file, indent=4)

        assert is_fmriprep_old_version(fmriprep_temp_dir)

        dataset_description = {"GeneratedBy": {"Name": "fMRIPrep",
                                               "Version": "24.0.0"},}

        with open(os.path.join(fmriprep_temp_dir, "dataset_description.json"), "w") as json_file:
            json.dump(dataset_description, json_file, indent=4)

        assert not is_fmriprep_old_version(fmriprep_temp_dir)


def test_is_aroma_in_fmriprep_output():
    from cvrmap.utils.tools import is_aroma_in_fmriprep_output

    with tempfile.TemporaryDirectory() as temp_dir:

        fmriprep_temp_dir = os.path.join(temp_dir, "fmriprep")
        temp_func_dir = os.path.join(fmriprep_temp_dir, "sub-007", "func")
        os.makedirs(temp_func_dir, exist_ok=True)

        assert not is_aroma_in_fmriprep_output(fmriprep_temp_dir)

        with open(os.path.join(temp_func_dir, "sub-007_task-gas_AROMAnoiseICs.csv"), "w") as f:
            f.write("Some text")

        assert is_aroma_in_fmriprep_output(fmriprep_temp_dir)


def test_init_pipeline_derivatives():
    from cvrmap.utils.tools import init_pipeline_derivatives
    with tempfile.TemporaryDirectory() as temp_dir:
        derivatives = {}
        config = {}
        config["output_dir"] = os.path.join(temp_dir, "output_dir")
        config["pipeline_name"] = "dummyName"
        config["version"] = "dummyVersion"

        derivatives = init_pipeline_derivatives(derivatives, config)

        assert "cvrmap" in derivatives.keys()

        dataset_description_path = os.path.join(temp_dir, "output_dir", "dataset_description.json")
        assert os.path.isfile(dataset_description_path)
