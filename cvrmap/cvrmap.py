from bids import BIDSLayout

from cvrmap.utils.tools import parse_args
from cvrmap.utils.loaders import load_derivatives, load_config
from cvrmap.preprocessing.denoising import denoising
from cvrmap.processing.probe import compute_probe_regressor
from cvrmap.processing.cvr import compute_cvr
from cvrmap.utils.tools import (init_pipeline_derivatives,
                                generate_report)

def main():
    args = parse_args()
    config = load_config(args)
    derivatives = load_derivatives(config)
    derivatives = init_pipeline_derivatives(derivatives, config)

    layout = BIDSLayout(config["bids_dir"], derivatives=derivatives)

    if config["analysis_level"] == "participant":
        participant_analysis_pipeline(layout, config)
    else:
        print(f"There is no group-level analysis for this software.")

    return None


def participant_analysis_pipeline(layout, config):

    subject_list = layout.derivatives["fMRIPrep"].get_subject()
    print(f"Found {len(subject_list)} subjects in fMRIPrep derivatives.")

    for sub in subject_list:
        probe_data_path, probe_metadata_path = compute_probe_regressor(layout, sub, config)
        denoising(layout, sub, config)
        compute_cvr(layout, sub, config)
        generate_report(layout, sub, config)
    return None