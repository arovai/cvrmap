import os
import tempfile
import pytest
import yaml
import json
from argparse import Namespace
import argparse

def test_load_derivatives():
    from cvrmap.utils.loaders import load_derivatives

    with tempfile.TemporaryDirectory() as temp_dir:

        fmriprep_temp_dir = os.path.join(temp_dir, "fmriprep")

        # Check when config derivatives is unset and it tries to guess fmriprep dir

        dummy_bids_dir = "dummy_name"
        config = {"bids_dir": dummy_bids_dir}

        with pytest.raises(FileNotFoundError, match=f"fMRIPrep directory not found at {dummy_bids_dir}/fmriprep."):
                load_derivatives(config)

        # Check when config derivatives is malformed

        config = {"derivatives": ["key value"]}
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid format for -d/--derivatives:"
                                                             " 'key value' must be in 'key=value' format."):
            load_derivatives(config)

        # Check when fmriprep value is not set at all

        config = {"derivatives": ["key=value"]}
        with pytest.raises(ValueError, match="You must specify the directory where fMRIPrep outputs are stored, "
                                             "using e.g. '--derivatives fmriprep=/path/to/fmriprep'."):
            load_derivatives(config)

        # Check when there is no folder at all

        config = {"derivatives": [f"fmriprep={fmriprep_temp_dir}"]}

        with pytest.raises(FileNotFoundError, match=f"fMRIPrep directory not found at {fmriprep_temp_dir}."):
            load_derivatives(config)

        # Check when there is no dataset_description.json
        os.mkdir(fmriprep_temp_dir)

        with pytest.raises(FileNotFoundError, match=f"Dataset description file for fMRIPrep not found. "
                                                    f"Are you sure your fMRIPrep directory is {fmriprep_temp_dir}?"):
            load_derivatives(config)

        # Check when dataset_description.json has wrong pipeline name
        pipeline_name = "DummyName"
        dataset_description = {"GeneratedBy": {"Name": pipeline_name}}

        with open(os.path.join(fmriprep_temp_dir, "dataset_description.json"), "w") as json_file:
            json.dump(dataset_description, json_file, indent=4)

        with pytest.raises(ValueError, match=f"Wrong dataset description for fMRIPrep output: found {pipeline_name},"
                                             f"expected 'fMRIPrep'"):
            load_derivatives(config)

        # Check when there are no preprocessed subjects with old version
        dataset_description = {"GeneratedBy": {"Name": "fMRIPrep",
                                               "Version": "21.0.4"},}

        with open(os.path.join(fmriprep_temp_dir, "dataset_description.json"), "w") as json_file:
            json.dump(dataset_description, json_file, indent=4)

        with pytest.raises(ValueError, match=f"No subject directory found at fMRIPrep output path {fmriprep_temp_dir}"):
            load_derivatives(config)

        # Check when ICA-AROMA not present in outputs for old version
        temp_func_dir = os.path.join(fmriprep_temp_dir, "sub-007", "func")
        os.makedirs(temp_func_dir, exist_ok=True)

        with open(os.path.join(temp_func_dir, "sub-007_task-gas_AROMAnoiseICs.csv"), "w") as f:
            f.write("Some text")

        derivatives = load_derivatives(config)

        assert derivatives["fmriprep"] == fmriprep_temp_dir

        # Check when fmripost-aroma is not specified with newer version
        dataset_description = {"GeneratedBy": {"Name": "fMRIPrep",
                                               "Version": "25.0.0"}, }

        with open(os.path.join(fmriprep_temp_dir, "dataset_description.json"), "w") as json_file:
            json.dump(dataset_description, json_file, indent=4)

        with pytest.raises(ValueError, match="You are using outputs of fMRIPrep from a version that does not contain ICA-AROMA outputs."
                                             "You must therefore run 'fmripost-aroma' and specify the corresponding output directory"
                                             "using e.g. '--derivatives fmripost-aroma=/path/to/fmriprep'."):
            load_derivatives(config)

        # Now add fmripost-aroma in config

        fmripost_aroma_path = os.path.join(temp_dir, "fmripost-aroma")
        config["derivatives"].append(f"fmripost-aroma={fmripost_aroma_path}")

        with pytest.raises(FileNotFoundError, match=f"fmripost-aroma directory not found at {fmripost_aroma_path}."):
            load_derivatives(config)

        os.mkdir(fmripost_aroma_path)

        derivatives = load_derivatives(config)

        assert derivatives["fmriprep"] == fmriprep_temp_dir
        assert derivatives["fmripost-aroma"] == fmripost_aroma_path


def test_load_config_from_file():
    from cvrmap.utils.loaders import load_config_from_file
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "config.yaml")

        with pytest.raises(FileNotFoundError, match=f"Provided configuration file {config_file} not found."):
            load_config_from_file(config_file)

        config = {"key": "value"}
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        assert load_config_from_file(config_file) == {"key": "value"}


def test_load_config():
    from cvrmap.utils.loaders import load_config

    args = Namespace(argument1="value1", argument2="value2", config=None)
    config = load_config(args)
    config.pop("version")
    config.pop("pipeline_name")
    assert config == dict(argument1="value1", argument2="value2", config=None)

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "config.yaml")
        config = {"key": "value"}
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        args = Namespace(bids_dir="value1",
                         output_dir="value2",
                         analysis_level="value3",
                         config=config_file)
        config = load_config(args)
        config.pop("version")
        config.pop("pipeline_name")
        assert config == dict(bids_dir="value1",
                                         output_dir="value2",
                                         analysis_level="value3",
                                         key="value",
                                         config=config_file)
