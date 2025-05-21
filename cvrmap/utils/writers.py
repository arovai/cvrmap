import os
import json
import numpy as np
import matplotlib.pyplot as plt

from cvrmap.utils.bids import build_output_path

def write_dataset_description(output_path, config):
    os.makedirs(output_path, exist_ok=True)
    dataset_description = {
                            "Name": config["pipeline_name"],
                            "BIDSVersion": "1.6.0",
                            "PipelineDescription": {
                                "Name": config["pipeline_name"],
                                "Version": config["version"],
                                "CodeURL": "https://github.com/ln2t/CVRmap"
                                }
                            }

    dataset_description_path = os.path.join(output_path, "dataset_description.json")
    with open(dataset_description_path, "w") as f:
        json.dump(dataset_description, f, indent=4)

    return None


def write_timeseries_data_and_figure(layout,
                                     sub,
                                     name,
                                     description,
                                     data,
                                     config,
                                     sampling_frequency=None):

    output_data_path = build_output_path(layout,
                                    sub,
                                    config,
                                    entities=dict(desc=name,
                                                  suffix="timecourse",
                                                  extension=".tsv"))
    output_json_path = output_data_path.replace(".tsv", ".json")
    output_figure_path = output_data_path.replace(".tsv", ".png")

    if sampling_frequency is None:
        sampling_frequency = 1./float(config["t_r"])

    output_metadata = dict(Name=name,
                           Description=description,
                           SamplingFrequency=sampling_frequency,
                           Units="s")

    write_data_to_tsv(data, output_data_path)
    write_json(output_metadata, output_json_path)
    write_figure(data,
                 sampling_frequency,
                 output_figure_path,
                 title=description,
                 xlabel=f"Time ({output_metadata['Units']})",
                 ylabel=f"Arbitrary units (BOLD)")
    return output_data_path, output_json_path


def write_data_to_tsv(data, output_path):
    np.savetxt(output_path, data, delimiter='\t')
    print(f"Data saved at {output_path}")


def write_json(metadata_dict, output_path):
    print(f"Json file saved at {output_path}")
    json.dump(metadata_dict, output_path)


def write_figure(data, sampling_frequency, output_path, **kwargs):
    time_span = np.arange(len(data))/sampling_frequency
    plt.figure(figsize=(10, 6))
    plt.plot(data, time_span, marker='o', linestyle='-', color='b', **kwargs)
    plt.savefig(output_path)
    print(f"Figure saved at {output_path}")