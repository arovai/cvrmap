import os
import argparse
import json
from packaging.version import Version
from glob import glob

from cvrmap.utils.writers import write_dataset_description

def setup_terminal_colors():
    import warnings
    import traceback
    import sys

    # ANSI escape codes for colors
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    def custom_warning_format(message, category, filename, lineno, line=None):
        # Define the color for the warning message
        return f"{YELLOW}{filename}:{lineno}: {category.__name__}: {message}{RESET}\n"

    # Set the custom warning formatter
    warnings.formatwarning = custom_warning_format

    # ANSI escape codes for colors
    RED = '\033[91m'
    RESET = '\033[0m'

    def custom_exception_handler(exc_type, exc_value, exc_traceback):
        # Format the exception traceback with color
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"{RED}{tb_str}{RESET}", end="")

    # Set the custom exception handler
    sys.excepthook = custom_exception_handler


def get_version():
    """
        Print version from package

    Returns:
        str
    """

    import importlib.metadata

    version = importlib.metadata.version('cvrmap')

    return version


def parse_args():
    parser = argparse.ArgumentParser(
        description="CVRmap: Compute maps of Cerebrovascular Reactivity")

    parser.add_argument("bids_dir",
                        nargs="?",
                        type=str,
                        help="BIDS root directory containing the dataset.")
    parser.add_argument("output_dir",
                        nargs="?",
                        type=str,
                        help="Directory where to store the outputs.")
    parser.add_argument("analysis_level",
                        nargs="?",
                        choices=["participant", "group"],
                        help="Analysis level: either 'participant' or 'group'.")

    parser.add_argument("-d",
                        "--derivatives",
                        nargs="+",
                        help="Specify pre-computed derivatives as 'key=value' pairs (e.g., -d fmriprep=/path/to/fmriprep fmripost-aroma=/path/to/fmripost-aroma).")

    parser.add_argument("-p", "--participant_label", type=str,
                        help="Participant label to process (e.g., '01').")
    parser.add_argument("-s", "--session", type=str,
                        help="Session to process (e.g., '1').")
    parser.add_argument("-t", "--task", type=str,
                        help="Task to process (e.g., 'restingstate').")
    parser.add_argument("--space",
                        help="Name of the space to be used. Must be associated with fmriprep output. "
                             "Default: \'MNI152NLin2009cAsym\'."
                             "Also accepts resolution modifier "
                             "(e.g. \'MNI152NLin2009cAsym:res-2\') as in fmriprep options.")

    parser.add_argument("--denoising", type=str,
                        help="Specify denoising strategy (see documentation)")
    parser.add_argument("--roi",
                        nargs="+",
                        help="Specify Region(s) Of Interest in the form 'name=value'"
                             "Value is either an absolute path (e.g. "
                             "'/path/to/mask.nii.gz') to an existing file, or"
                             "a regex (e.g. '*desc-brain_mask.nii.gz')."
                             "See documentation for more information.")

    parser.add_argument("--skip_bids_validation",
                        help="Whether or not to perform BIDS dataset validation",
                        action='store_true')
    parser.add_argument("--sloppy",
                        help="Only for testing, computes a small part of the maps to save time. Off by default.",
                        action="store_true")

    parser.add_argument("-c", "--config", type=str,
                        help="Path to the configuration file.")

    return parser.parse_args()


def is_fmriprep_old_version(fmriprep_path):
    dataset_description_path = os.path.join(fmriprep_path, "dataset_description.json")
    if os.path.isfile(dataset_description_path):

        with open(dataset_description_path, "r") as json_file:
            dataset_description = json.load(json_file)

        pipeline_name = dataset_description["GeneratedBy"]["Name"]
        if pipeline_name == "fMRIPrep":
            return Version(dataset_description["GeneratedBy"]["Version"]) < Version("23.1.0")
        else:
            raise ValueError(f"Wrong dataset description for fMRIPrep output: found {pipeline_name},"
                             f"expected 'fMRIPrep'")

    else:
        raise FileNotFoundError(f"Dataset description file for fMRIPrep not found. "
                                f"Are you sure your fMRIPrep directory is {fmriprep_path}?")


def is_aroma_in_fmriprep_output(fmriprep_path):
    fmriprep_output_list = glob(os.path.join(fmriprep_path, "sub-*"))
    if len(fmriprep_output_list) == 0:
        raise ValueError(f"No subject directory found at fMRIPrep output path {fmriprep_path}")
    else:
        subject_output_list = glob(os.path.join(fmriprep_output_list[0], "func", "*"))
        for item in subject_output_list:
            if "AROMAnoiseICs" in item:
                return True

    return False


def update_new_dict_values(dict_to_update, dict_with_new_value):
    new_dict = dict_to_update.copy()
    for key in dict_with_new_value.keys():
        if not key in dict_to_update.keys():
            new_dict[key] = dict_with_new_value[key]
    return new_dict


def remove_none_values(dict_to_clean):
    new_dict = dict_to_clean.copy()
    for key in dict_to_clean.keys():
        if dict_to_clean[key] is None:
            dict_to_clean.pop(key)
    return new_dict


def init_pipeline_derivatives(derivatives, config):

    new_derivatives = derivatives.copy()
    output_dir = config["output_dir"]
    new_derivatives["cvrmap"] = output_dir
    write_dataset_description(output_dir, config)

    return new_derivatives


def generate_report():
    return None


def build_output_path(layout, sub, config, entities):
    bids_path = ""
    return bids_path