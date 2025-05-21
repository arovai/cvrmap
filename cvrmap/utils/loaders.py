import argparse
import json
import os
import pandas as pd
import numpy as np
import yaml
import nibabel as nib
from sqlalchemy.testing import metadata_fixture
from sqlalchemy.testing.suite.test_reflection import metadata

from cvrmap.utils.tools import (is_fmriprep_old_version,
                                is_aroma_in_fmriprep_output,
                                get_version,
                                remove_none_values,
                                update_new_dict_values)


def load_key_equal_value_to_dict(key_equal_value_list):
    output_dict = {}

    for item in key_equal_value_list:
        if '=' in item:
            key, value = item.split('=', 1)
            output_dict[key] = value
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid format for -d/--derivatives: '{item}' must be in 'key=value' format.")

    return output_dict


def load_derivatives(config):
    """Convert list of 'key=value' items into a dictionary."""

    derivatives_list = config.get("derivatives", None)

    if derivatives_list:
        derivatives_dict = load_key_equal_value_to_dict(derivatives_list)
    else:
        print(f"No derivatives specified - try to guess paths...")
        derivatives_dict = {"fmriprep": os.path.join(config["bids_dir"], "fmriprep"),
                            "fmripost-aroma": os.path.join(config["bids_dir"], "fmripost-aroma")}

    if not "fmriprep" in derivatives_dict.keys():
        raise ValueError("You must specify the directory where fMRIPrep outputs are stored, "
                         "using e.g. '--derivatives fmriprep=/path/to/fmriprep'.")

    if os.path.isdir(derivatives_dict["fmriprep"]):
        print(f"fMRIPrep directory found at {derivatives_dict['fmriprep']}.")
    else:
        raise FileNotFoundError(f"fMRIPrep directory not found at {derivatives_dict['fmriprep']}.")

    # Check fMRIPrep version and corresponding fmripost-aroma directory
    if is_fmriprep_old_version(derivatives_dict["fmriprep"]):
        if is_aroma_in_fmriprep_output(derivatives_dict["fmriprep"]):
            print("fMRIPrep outputs contain ICA-AROMA results.")
        else:
            raise ValueError(f"fMRIPrep outputs does not appear to have ICA-AROMA outputs."
                             f"Please re-run fMRIPrep with '--use_aroma', or use the 'fmripost-aroma' tool.")
    else:
        if "fmripost-aroma" in derivatives_dict.keys():
            if os.path.isdir(derivatives_dict["fmripost-aroma"]):
                print(f"fmripost-aroma directory found at {derivatives_dict['fmripost-aroma']}")
            else:
                raise FileNotFoundError(f"fmripost-aroma directory not found at {derivatives_dict['fmripost-aroma']}.")
        else:
            raise ValueError("You are using outputs of fMRIPrep from a version that does not contain ICA-AROMA outputs."
                             "You must therefore run 'fmripost-aroma' and specify the corresponding output directory"
                             "using e.g. '--derivatives fmripost-aroma=/path/to/fmriprep'.")

    return derivatives_dict


def load_default_config():
    config = {}

    # CLI options
    config["participant_label"] = None
    config["task"] = None
    config["session"] = None
    config["space"] = "MNI152NLin2009cAsym"

    config["derivatives"] = []
    config["denoising"] = "refined_ica_aroma"
    config["roi"] = []

    config["sloppy"] = False
    config["skip_bids_validation"] = False

    # Non-CLI options
    config["pipeline_name"] = "CVRmap"
    config["version"] = get_version()

    # Processing options
    config["physio_smoothing_sigma"] = 0.06  # In units of seconds
    config["sigma_for_peak_localisation"] = 0.8  # In units of seconds

    return config


def load_config(args):
    """Load config for pipeline"""

    # Load config from cli
    cli_config = vars(args)

    # Load config from file, if provided
    file_config = {}
    if args.config:
        print(f"Loading configuration from {args.config}. "
              f"Beware that CLI options overwrite file configs.")
        file_config = load_config_from_file(args.config)

    # CLI has priority on config file
    config = update_new_dict_values(file_config, cli_config)

    # Fill remaining values with defaults
    config = remove_none_values(config)
    config = update_new_dict_values(config, load_default_config())

    # Convert roi list to dict
    if config["roi"]:
        if len(config["roi"]) > 1:
            raise ValueError(f"Only one ROI can be specified with --roi."
                             f"You set {config['roi']}.")
        else:
            config["roi"] = load_key_equal_value_to_dict(config["roi"])

    return config


def load_config_from_file(config_file):
    """Load config from yaml file"""
    if os.path.isfile(config_file):
        with open(config_file, "r") as cf:
            return yaml.safe_load(cf)
    else:
        raise FileNotFoundError(f"Provided configuration file {config_file} not found.")


def load_brain_mask(layout, sub, config):
    file_matches = layout.derivatives["fMRIPrep"].get(subject=sub,
                                       space=config["space"],
                                       task=config["task"],
                                       session=config["session"],
                                       return_type="filename",
                                       desc="brain",
                                       suffix="mask",
                                       extension=".nii.gz")
    if len(file_matches) == 0:
        raise ValueError(f"No brain mask found for subject {sub}")
    elif len(file_matches) > 1:
        raise ValueError(f"More than one brain mask found for subject {sub}")
    else:
        return file_matches[0]


def load_roi_mask(layout, sub, config):
    roi_name = config["roi"].keys()[0]
    roi_str = config["roi"][roi_name]
    if "/" in roi_str:
        if os.path.isfile(roi_str):
            return roi_name, roi_str
        else:
            raise FileNotFoundError(f"ROI file {roi_str} not found.")
    else:
        # Trying to interpret roi_str as a regex
        file_matches = layout.derivatives["fMRIPrep"].get(subject=sub,
                                           space=config["space"],
                                           task=config["task"],
                                           session=config["session"],
                                           return_type="filename",
                                           extension=".nii.gz",
                                           suffix="mask")

        file_matches = find_regex_match_in_list(roi_str, file_matches)

        if len(file_matches) == 0:
            raise ValueError(f"No match found for roi regex {roi_str}")
        elif len(file_matches) > 1:
            raise ValueError(f"More than one match found for roi regex {roi_str},"
                             f"please refined the expression to have exactly one"
                             f"match.")
        else:
            return roi_name, file_matches[0]


def find_regex_match_in_list(regex_str, str_list):
    output_list = []
    for item in str_list:
        if regex_str in item:  # TODO: adapt this to have true regex match
            output_list.append(item)
    return output_list


def load_tr_from_nifti(img_path):
    _, _, _, t_r = nib.load(img_path).get_zooms()
    return t_r


def load_bold_img(layout, sub, config):
    file_matches = layout.derivatives["fMRIPrep"].get(subject=sub,
                                       task=config["task"],
                                       session=config["session"],
                                       space=config["space"],
                                       return_type="filename",
                                       extension=".nii.gz",
                                       suffix="bold")

    if len(file_matches) == 0:
        raise ValueError(f"No bold timeseries found for subject {sub}")
    elif len(file_matches) > 1:
        raise ValueError(f"More than one bold timeseries found for subject {sub}")
    else:
        bold_path = file_matches[0]
        if os.path.isfile(bold_path):
            config["t_r"] = load_tr_from_nifti(bold_path)
            return bold_path, config
        else:
            raise FileNotFoundError(f"BOLD file {bold_path} not found.")


def load_physio_data(layout, sub, config):
    file_matches = layout.get(subject=sub,
                              return_type="filename",
                              extension=".tsv.gz",
                              task=config["task"],
                              session=config["session"],
                              suffix="physio")

    if len(file_matches) == 0:
        raise FileNotFoundError(f"No physiological data found for subject {sub}.")
    elif len(file_matches) > 1:
        raise ValueError(f"More than one match found for the physiological data"
                         f"of subject {sub}, please review your configuration.")
    else:
        data_path = file_matches[0]
        metadata_path = data_path.replace(".tsv.gz", ".json")
        if os.path.isfile(metadata_path):
            return data_path, metadata_path
        else:
            raise FileNotFoundError(f"Metadata for physiological data for subject {sub}"
                                    f"not found.")


def load_co2_data(layout, sub, config):
    physio_data_path, physio_metadata_path = load_physio_data(layout, sub, config)
    physio_metadata = json.load(physio_metadata_path)
    physio_cols = physio_metadata["Columns"]

    if "co2" in physio_cols:
        co2_idx = physio_cols.index("co2")
    else:
        raise ValueError(f"Physio metadata do not appear"
                         f"to have 'co2' readings, check your data.")

    physio_data = pd.read_csv(physio_data_path, sep='\t', header=False)
    co2_data = physio_data.loc[co2_idx]

    co2_metadata = {
        "SamplingFrequency": physio_metadata["SamplingFrewuency"],
        "Units": physio_metadata["co2"]["Units"]
    }

    return co2_data, co2_metadata