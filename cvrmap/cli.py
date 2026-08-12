import argparse
import json
import os
import textwrap
from pathlib import Path

from . import __version__

# Set matplotlib backend to non-GUI to avoid tkinter issues in parallel processing
import matplotlib

matplotlib.use("Agg")


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class ColoredHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter with colored section headers."""

    def __init__(self, prog, indent_increment=2, max_help_position=40, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = f"{Colors.BOLD}Usage:{Colors.END} "
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        if heading:
            heading = f"{Colors.BOLD}{Colors.CYAN}{heading}{Colors.END}"
        super().start_section(heading)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    description = textwrap.dedent(
        f"""
        {Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
        ║                             CVRMAP v{__version__:<10}                              ║
        ║               Cerebrovascular Reactivity Mapping Pipeline                     ║
        ╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

        {Colors.BOLD}Description:{Colors.END}
          cvrmap performs BIDS-compliant CVR analysis on fMRI and probe data,
          generating derivative maps, summary statistics, and report outputs.

        {Colors.BOLD}Workflow:{Colors.END}
          1. Discover participant/task data from BIDS and derivatives
          2. Load and validate BOLD and probe inputs
          3. Run preprocessing, cross-correlation, and delay estimation
          4. Compute CVR maps and derived coefficient outputs
          5. Write BIDS-compliant derivatives and reports
        """
    )

    epilog = textwrap.dedent(
        f"""
        {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
        {Colors.BOLD}EXAMPLES{Colors.END}
        {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

          {Colors.YELLOW}# Basic participant-level run{Colors.END}
          cvrmap /data/bids /data/output participant --task gas

          {Colors.YELLOW}# Specify fMRIPrep derivatives location{Colors.END}
          cvrmap /data/bids /data/output participant --task gas \\
              --derivatives fmriprep=/data/derivatives/fmriprep

          {Colors.YELLOW}# Run ROI-probe mode using built-in SSS mask{Colors.END}
          cvrmap /data/bids /data/output participant --task gas \\
              --roi-probe --roi-mask SSS

          {Colors.YELLOW}# Enable verbose logging and custom config{Colors.END}
          cvrmap /data/bids /data/output participant --task gas \\
              --config custom.yaml --verbose

        Documentation:  https://github.com/arovai/cvrmap
        Version:        {__version__}
        """
    )

    parser = argparse.ArgumentParser(
        prog="cvrmap",
        description=description,
        epilog=epilog,
        formatter_class=ColoredHelpFormatter,
        add_help=False,
    )

    required = parser.add_argument_group(f"{Colors.BOLD}Required Arguments{Colors.END}")
    required.add_argument(
        "bids_dir",
        type=Path,
        metavar="INPUT_DIR",
        help="Path to the BIDS dataset root directory.",
    )
    required.add_argument(
        "output_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Path to output directory for CVR derivatives.",
    )
    required.add_argument(
        "analysis_level",
        choices=["participant"],
        metavar="{participant}",
        help="Analysis level. Currently only 'participant' is supported.",
    )

    general = parser.add_argument_group(f"{Colors.BOLD}General Options{Colors.END}")
    general.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    general.add_argument(
        "--version",
        action="version",
        version=f"cvrmap {__version__}",
        help="Show program version and exit.",
    )
    general.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging).",
    )
    general.add_argument(
        "--debug-level",
        "--debug_level",
        type=int,
        choices=[0, 1],
        default=None,
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="FILE",
        help="Path to configuration file (.json, .yaml, or .yml). CLI arguments override config values.",
    )

    derivatives = parser.add_argument_group(f"{Colors.BOLD}Input Derivatives Options{Colors.END}")
    derivatives.add_argument(
        "-d",
        "--derivatives",
        action="append",
        nargs="+",
        metavar="NAME=PATH",
        dest="derivatives",
        help=(
            "Specify location of BIDS derivatives. Format: name=path "
            "(e.g., fmriprep=/data/derivatives/fmriprep). Can be specified multiple times "
            "or with multiple values after one flag."
        ),
    )

    filters = parser.add_argument_group(
        f"{Colors.BOLD}BIDS Entity Filters{Colors.END}",
        "Filter which data to process based on BIDS entities.",
    )
    filters.add_argument(
        "-p",
        "--participant-label",
        "--participant_label",
        metavar="LABEL",
        dest="participant_label",
        nargs="+",
        help="Process one or more participants (without 'sub-' prefix).",
    )
    filters.add_argument(
        "-t",
        "--task",
        metavar="TASK",
        help="Process only this task (without 'task-' prefix).",
    )
    filters.add_argument(
        "--space",
        metavar="SPACE",
        help="Process only data in this template space (e.g., MNI152NLin2009cAsym).",
    )

    processing = parser.add_argument_group(f"{Colors.BOLD}Processing Options{Colors.END}")
    processing.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        metavar="N",
        help="Number of parallel jobs for voxel processing. -1 uses all CPUs (default: -1).",
    )
    processing.add_argument(
        "--baseline-method",
        type=str,
        choices=["peakutils", "mean"],
        default=None,
        metavar="METHOD",
        help="Probe baseline method: peakutils (default) or mean.",
    )

    roi = parser.add_argument_group(f"{Colors.BOLD}ROI Probe Options{Colors.END}")
    roi.add_argument(
        "--roi-probe",
        action="store_true",
        help="Enable ROI-based probe instead of physiological recordings.",
    )
    roi.add_argument(
        "--roi-coordinates",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="ROI coordinates in millimeters (world space), e.g. --roi-coordinates 0 -52 26.",
    )
    roi.add_argument(
        "--roi-radius",
        type=float,
        default=6.0,
        metavar="MM",
        help="Radius in millimeters for spherical ROI (default: 6.0).",
    )
    roi.add_argument(
        "--roi-mask",
        type=str,
        metavar="PATH_OR_KEYWORD",
        help="Path/pattern to binary ROI mask, or 'SSS' for built-in Superior Sagittal Sinus mask.",
    )
    roi.add_argument(
        "--roi-atlas",
        type=str,
        metavar="PATH",
        help="Path to atlas NIfTI file for ROI extraction.",
    )
    roi.add_argument(
        "--roi-region-id",
        type=int,
        metavar="ID",
        help="Region ID/label in atlas for ROI extraction.",
    )
    roi.add_argument(
        "--roi-label",
        type=str,
        metavar="LABEL",
        help="Label for ROI outputs in BIDS naming. Required for custom --roi-mask paths/patterns.",
    )

    temporal = parser.add_argument_group(f"{Colors.BOLD}ROI Probe Filter Options{Colors.END}")
    temporal.add_argument(
        "--probe-bandpass-filter",
        action="store_true",
        help="Enable bandpass filtering for ROI probe signal (requires --roi-probe).",
    )
    temporal.add_argument(
        "--probe-highpass",
        type=float,
        metavar="HZ",
        help="Highpass cutoff frequency for ROI probe filter in Hz.",
    )
    temporal.add_argument(
        "--probe-lowpass",
        type=float,
        metavar="HZ",
        help="Lowpass cutoff frequency for ROI probe filter in Hz.",
    )

    return parser


def parse_derivatives_arg(derivatives_list):
    """Parse derivatives list of NAME=PATH entries into a dictionary."""
    if not derivatives_list:
        return {}

    flattened = []
    for item in derivatives_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    derivatives_dict = {}
    for derivative_arg in flattened:
        if "=" not in derivative_arg:
            raise ValueError(
                f"Invalid derivatives argument: {derivative_arg}. "
                "Expected format: name=path (e.g., fmriprep=/path/to/data)."
            )

        name, path = derivative_arg.split("=", 1)
        if not name or not path:
            raise ValueError(
                f"Invalid derivatives argument: {derivative_arg}. "
                "Name or path is missing."
            )
        derivatives_dict[name] = Path(path)

    return derivatives_dict


def is_safe_within_directory(base_dir, target_path):
    """Return True when target_path resolves within base_dir."""
    base_real = os.path.realpath(base_dir)
    target_real = os.path.realpath(target_path)
    try:
        return os.path.commonpath([base_real, target_real]) == base_real
    except ValueError:
        return False


def is_derivative_dataset(path):
    base_dir = os.path.abspath(path)
    description_path = os.path.abspath(os.path.join(base_dir, "dataset_description.json"))
    if not is_safe_within_directory(base_dir, description_path):
        return False
    if not os.path.isfile(description_path):
        return False

    try:
        with open(description_path, "r", encoding="utf-8") as description_file:
            description = json.load(description_file)
    except (OSError, json.JSONDecodeError):
        return False

    dataset_type = str(description.get("DatasetType", "")).lower()
    return dataset_type == "derivative"


def validate_derivatives(derivatives, logger):
    """Warn for non-directory derivative entries."""
    for name, path in derivatives.items():
        if not path.is_dir():
            logger.warning(f"Derivatives path does not exist or is not a directory: '{path}' for pipeline '{name}'.")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.debug_level is not None:
        args.verbose = args.debug_level == 1

    if not args.bids_dir.is_dir():
        parser.error(f"INPUT_DIR '{args.bids_dir}' does not exist or is not a directory.")

    from .logger import setup_logging

    logger = setup_logging(verbose=args.verbose, logger_name="cvrmap.cli")

    try:
        derivatives = parse_derivatives_arg(args.derivatives)
    except ValueError as exc:
        parser.error(str(exc))
    validate_derivatives(derivatives, logger)

    args.bids_dir = str(args.bids_dir)
    args.output_dir = str(args.output_dir)
    args.config = str(args.config) if args.config else None
    args.derivatives = [f"{name}={path}" for name, path in derivatives.items()]

    logger.info(f"bids_dir: {args.bids_dir}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"analysis_level: {args.analysis_level}")
    logger.info(f"participant_label: {args.participant_label}")
    logger.info(f"task: {args.task}")
    logger.info(f"derivatives: {args.derivatives}")

    fmriprep_dir = None
    args.direct_fmriprep_input = False

    fmriprep_entry = derivatives.get("fmriprep")
    if fmriprep_entry:
        if fmriprep_entry.is_dir():
            fmriprep_dir = str(fmriprep_entry)
        else:
            logger.warning(f"fmriprep path specified but does not exist: {fmriprep_entry}")

    if not fmriprep_dir:
        if args.roi_probe and is_derivative_dataset(args.bids_dir):
            fmriprep_dir = args.bids_dir
            args.direct_fmriprep_input = True
            logger.info("Using bids_dir as the fMRIPrep derivative root for ROI probe mode")
        else:
            fmriprep_dir = os.path.join(args.output_dir, "fmriprep")
            if not os.path.isdir(fmriprep_dir):
                logger.warning(f"fmriprep derivatives not specified and default path does not exist: {fmriprep_dir}")
                parser.error(
                    "fmriprep derivatives not found. Please specify with "
                    "--derivatives fmriprep=/path/to/fmriprep/derivatives or ensure "
                    f"{fmriprep_dir} exists."
                )
    logger.info(f"fmriprep_dir: {fmriprep_dir}")

    from .io import process_config

    config = process_config(user_config_path=args.config)
    config["n_jobs"] = args.n_jobs

    if args.baseline_method:
        if "physio" not in config:
            config["physio"] = {}
        config["physio"]["baseline_method"] = args.baseline_method
        logger.info(f"Baseline method set to: {args.baseline_method}")

    if args.task and args.task.lower() in {"restingstate", "resting-state", "rest"}:
        current_baseline_method = config.get("physio", {}).get("baseline_method", "peakutils")
        if current_baseline_method == "peakutils":
            logger.warning(
                "Task appears to be resting-state data. Consider using --baseline-method mean "
                "for better baseline estimation in resting-state data without gas challenge."
            )

    if args.roi_probe:
        logger.info("ROI probe mode enabled via command line")
        if "roi_probe" not in config:
            config["roi_probe"] = {}

        config["roi_probe"]["enabled"] = True

        if args.roi_coordinates:
            config["roi_probe"]["method"] = "coordinates"
            config["roi_probe"]["coordinates_mm"] = list(args.roi_coordinates)
            config["roi_probe"]["radius_mm"] = args.roi_radius
            logger.info(f"ROI coordinates: {args.roi_coordinates}, radius: {args.roi_radius}mm")
        elif args.roi_mask:
            config["roi_probe"]["method"] = "mask"
            if args.roi_mask.upper() == "SSS":
                import pkg_resources

                sss_mask_path = pkg_resources.resource_filename("cvrmap", "data/SuperiorSagittalSinus_mask.nii.gz")
                config["roi_probe"]["mask_path"] = sss_mask_path
                config["roi_probe"]["label"] = "SSS"
                logger.info(f"Using built-in Superior Sagittal Sinus (SSS) mask: {sss_mask_path}")
            else:
                if not args.roi_label:
                    parser.error(
                        "When using --roi-mask with a custom path or pattern, --roi-label is required "
                        "to identify outputs in BIDS naming."
                    )
                config["roi_probe"]["mask_path"] = args.roi_mask
                config["roi_probe"]["label"] = args.roi_label
                logger.info(f"ROI mask: {args.roi_mask}, label: {args.roi_label}")
        elif args.roi_atlas and args.roi_region_id is not None:
            config["roi_probe"]["method"] = "atlas"
            config["roi_probe"]["atlas_path"] = args.roi_atlas
            config["roi_probe"]["region_id"] = args.roi_region_id
            logger.info(f"ROI atlas: {args.roi_atlas}, region: {args.roi_region_id}")
        else:
            parser.error(
                "When using --roi-probe, you must specify either:\n"
                "  --roi-coordinates X Y Z (and optionally --roi-radius)\n"
                "  --roi-mask PATH\n"
                "  --roi-atlas PATH --roi-region-id ID"
            )

    if args.probe_bandpass_filter:
        if not args.roi_probe:
            parser.error("--probe-bandpass-filter can only be used with --roi-probe mode")

        highpass_hz = args.probe_highpass if args.probe_highpass is not None else 0.02
        lowpass_hz = args.probe_lowpass if args.probe_lowpass is not None else 0.04

        if "roi_probe" not in config:
            config["roi_probe"] = {}

        config["roi_probe"]["bandpass_filter"] = {
            "enabled": True,
            "highpass": highpass_hz,
            "lowpass": lowpass_hz,
        }
        logger.info(f"ROI probe bandpass filter enabled: highpass={highpass_hz} Hz, lowpass={lowpass_hz} Hz")
    elif config.get("roi_probe", {}).get("bandpass_filter", {}).get("enabled", False):
        bp_config = config["roi_probe"]["bandpass_filter"]
        if bp_config.get("highpass") is None and bp_config.get("lowpass") is None:
            logger.warning("Bandpass filter enabled in config but no cutoff frequencies specified. Disabling filter.")
            config["roi_probe"]["bandpass_filter"]["enabled"] = False
        else:
            filter_desc = []
            if bp_config.get("highpass") is not None:
                filter_desc.append(f"highpass={bp_config['highpass']} Hz")
            if bp_config.get("lowpass") is not None:
                filter_desc.append(f"lowpass={bp_config['lowpass']} Hz")
            logger.info(f"ROI probe bandpass filter from config: {', '.join(filter_desc)}")

    from .pipeline import Pipeline

    pipeline = Pipeline(args, logger, fmriprep_dir, config=config)
    pipeline.run()

if __name__ == "__main__":
    main()
