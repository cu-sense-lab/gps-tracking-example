import os
import logging
from pathlib import Path
import yaml
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dataclasses import dataclass
from .sample_streaming import SampleParameters

@dataclass
class BandConfiguration:
    center_freq: float
    inter_freq: Optional[float] = None

    @staticmethod
    def from_dict(band_dict: Dict[str, Any]) -> "BandConfiguration":
        if "center_freq" not in band_dict:
            raise ValueError("Band configuration dictionary must contain 'center_freq' key.")
        center_freq = band_dict["center_freq"]
        inter_freq = band_dict.get("inter_freq")
        return BandConfiguration(center_freq=center_freq, inter_freq=inter_freq)

@dataclass
class ChannelConfiguration:
    samp_rate: float
    band_ids: List[str]
    sample_params: SampleParameters

def parse_channel_configurations_from_dict(config_dict: Dict[str, Dict[str, Any]], verbose: bool = False) -> Dict[str, ChannelConfiguration]:
    """
    NOTE: Receiver configurations can optionally specify an "inherit" key to indicate that they should inherit and override parameters from a parent configuration. For example:

    "receiver_configurations": {
        "base_config": {
            "samp_rate": 12500000,
            "bands": ["band1", "band2"],
            "sample_format": {
                "is_i_lsb": True,
                "is_integer": True,
                "is_signed": True,
                "is_complex": True,
                "bit_depth": 16,
            },
        },
        "collect1_config": {
            "inherit": "base_config",
            "bands": ["band1"],  # override bands to only include band1
        },
    }

    NOTE: the base config may fail to parse if it doesn't specify all required fields.  If a config parse fails, we omit it from
    the parsed configurations.  By default, we will not warn the user (so they don't see warnings due to base/default configs failing),
    but warnings may be turned on for debug purposes using the `verbose` flag.
    """
    channel_configs = {}
    for channel_config_id, channel_config_dict in config_dict.items():
        if "inherit" in channel_config_dict:
            parent_id = channel_config_dict["inherit"]
            if parent_id not in config_dict:
                raise ValueError(f"Parent receiver configuration ID '{parent_id}' not found in channel configurations metadata.")
            parent_config: Dict[str, Any] = config_dict[parent_id]
            merged_config = parent_config.copy()
            merged_config.update(channel_config_dict)
            del merged_config["inherit"]
            channel_config_dict = merged_config
        try:
            if "samp_rate" not in channel_config_dict:
                raise ValueError(f"Channel configuration '{channel_config_id}' is missing 'samp_rate' key.")
            if "bands" not in channel_config_dict:
                raise ValueError(f"Channel configuration '{channel_config_id}' is missing 'bands' key.")
            if "sample_format" not in channel_config_dict:
                raise ValueError(f"Channel configuration '{channel_config_id}' is missing 'sample_format' key.")
            samp_rate = channel_config_dict["samp_rate"]
            band_ids = sorted(list(channel_config_dict["bands"]))
            sample_format_dict = channel_config_dict["sample_format"]
            sample_params = SampleParameters.from_dict(sample_format_dict)
            channel_configs[channel_config_id] = ChannelConfiguration(samp_rate=samp_rate, band_ids=band_ids, sample_params=sample_params)
        except Exception as e:
            if verbose:
                logging.warning(e)
    return channel_configs

@dataclass
class CollectMetadata:
    channel_config_id: str
    filename: str
    notes: Optional[str] = None

    @staticmethod
    def from_dict(collect_dict: Dict[str, Any]) -> "CollectMetadata":
        if "channel_config" not in collect_dict:
            raise ValueError("Collect metadata dictionary must contain 'channel_config' key.")
        if "filename" not in collect_dict:
            raise ValueError("Collect metadata dictionary must contain 'filename' key.")
        channel_config_id = collect_dict["channel_config"]
        filename = collect_dict["filename"]
        notes = collect_dict.get("notes")
        return CollectMetadata(channel_config_id=channel_config_id, filename=filename, notes=notes)

@dataclass
class ExperimentMetadata:
    band_configurations: Dict[str, BandConfiguration]
    channel_configurations: Dict[str, ChannelConfiguration]
    collects: Dict[str, CollectMetadata]
    notes: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None

    @staticmethod
    def from_dict(metadata_dict: Dict[str, Any], verbose: bool = False) -> "ExperimentMetadata":
        if "band_configurations" not in metadata_dict:
            raise ValueError("Metadata dictionary must contain 'band_configurations' key.")
        if "channel_configurations" not in metadata_dict:
            raise ValueError("Metadata dictionary must contain 'channel_configurations' key.")
        if "collections" not in metadata_dict:
            raise ValueError("Metadata dictionary must contain 'collections' key.")
        band_configurations = {band_id: BandConfiguration.from_dict(band_dict) for band_id, band_dict in metadata_dict["band_configurations"].items()}
        channel_configurations = parse_channel_configurations_from_dict(metadata_dict["channel_configurations"], verbose)
        collects = {collect_id: CollectMetadata.from_dict(collect_dict) for collect_id, collect_dict in metadata_dict["collections"].items()}
        return ExperimentMetadata(band_configurations=band_configurations, channel_configurations=channel_configurations, collects=collects)


    @property
    def collect_ids(self) -> List[str]:
        return sorted(list(self.collects.keys()))

    @property
    def channel_ids(self) -> List[str]:
        return sorted(list(self.channel_configurations.keys()))
    
    @property
    def band_ids(self) -> List[str]:
        return sorted(list(self.band_configurations.keys()))
    
def parse_experiment_metadata(metadata_dict: Dict[str, Any], verbose: bool = False) -> ExperimentMetadata:
    """
    Retrieve the receiver configuration given a metadata dictionary.

    metadata dictionary has the following structure:

    {
        "collections": {
            "<collect_id>": {
                "rx_config": "<rx_config_id>",
                "filename": "<data_filename>",
                ...}
            }
        "band_configurations": {
            "<band_id>": {
                "center_freq": ...,
                "inter_freq": ...,
            }
        },
        "receiver_configurations": {
            "<rx_config_id>": {
                "samp_rate": 12500000,
                "bands": [...],
                "sample_format": {
                    "is_i_lsb": ...,
                    "is_integer": ...,
                    "is_signed": ...,
                    "is_complex": ...,
                    "bit_depth": ...,
                },
            }
        

    Args:
        metadata_dict (Dict[str, Any]): The metadata dictionary containing collect information.
        collect_name (str): The name of the collect.

    Returns:
        Dict[str, Dict[str, Any]]: The receiver configuration dictionaries for each collect.
    """
    return ExperimentMetadata.from_dict(metadata_dict, verbose=verbose)

def print_experiment_available_collects_and_bands(metadata: ExperimentMetadata, print_prefix: str = ""):
    # print(print_prefix + f"Bands: " + " ".join(metadata.band_ids))
    # print(print_prefix + f"Channel configs: " + " ".join(metadata.channel_ids))
    print(print_prefix + f"Available bands: " + " ".join(metadata.band_ids))
    print(print_prefix + f"Available collects:")
    for collect_id in metadata.collect_ids:
        collect = metadata.collects[collect_id]
        channel_config_id = collect.channel_config_id
        band_ids = metadata.channel_configurations[channel_config_id].band_ids
        print(print_prefix + f"  {collect_id}: " + " ".join(band_ids))

def load_experiment_metadata_from_file(metadata_filepath: str, print_summary: bool = False, print_prefix: str = "", verbose_parsing: bool = False) -> ExperimentMetadata:
    """
    Load the experiment metadata from a YAML file.

    Args:
        metadata_filepath (str): The file path to the metadata YAML file.
    Returns:
        ExperimentMetadata: The parsed experiment metadata.
    """
    with open(metadata_filepath, "r") as f:
        metadata_dict = yaml.safe_load(f)
    metadata = parse_experiment_metadata(metadata_dict, verbose=verbose_parsing)
    if print_summary:
        print_experiment_available_collects_and_bands(metadata, print_prefix)
    return metadata


import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes

def plot_receiver_channel_bands(
        fig: Figure | SubFigure,
        metadata: ExperimentMetadata,
        include_bands: Optional[Iterable[str]] = None,
        exclude_bands: Optional[Iterable[str]] = None,
        samp_bandwidth_height: float = 0.5,
        samp_center_height: float = 0.8,
):
    axes = fig.subplots(1, 2, sharey=True)
    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    # Want to plot two axes;
    # Left shows baseband frequencies and right shows RF frequencies
    # Channel IDs go down the y-axis
    band_colors = {}
    for i, band_id in enumerate(metadata.band_ids):
        band_colors[band_id] = f"C{i}"
    
    band_configurations = metadata.band_configurations
    for i, channel_id in enumerate(metadata.channel_ids):
        channel_config = metadata.channel_configurations[channel_id]
        samp_bandwidth_MHz = channel_config.samp_rate / 1e6
        is_real = not channel_config.sample_params.is_complex
        if is_real:
            # Only shade positive frequencies
            ax1.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], 0, samp_bandwidth_MHz / 2, color="r", alpha=0.3)
        else:
            ax1.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], -samp_bandwidth_MHz / 2, samp_bandwidth_MHz / 2, color="b", alpha=0.3)
        
        for band_id in channel_config.band_ids:
            band_config = band_configurations[band_id]
            baseband_if_MHz = band_config.inter_freq / 1e6
            rf_center_MHz = band_config.center_freq / 1e6

            # Plot delta-like markers at IF (baseband) and RF center frequencies.
            ax1.vlines(baseband_if_MHz, i - samp_center_height / 2, i + samp_center_height / 2, color="k", linewidth=2)
            # ax1.plot([baseband_if], [i], marker="|", markersize=12, color="C0")

            ax2.vlines(rf_center_MHz, i - samp_center_height / 2, i + samp_center_height / 2, color="k", linewidth=2)
            # ax2.plot([rf_center], [i], marker="|", markersize=12, color="C1")
            
            rf_band_center_MHz = (band_config.center_freq - band_config.inter_freq) / 1e6
            if is_real:
                ax2.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], rf_band_center_MHz, rf_band_center_MHz + samp_bandwidth_MHz / 2, color="r", alpha=0.3)
            else:
                ax2.fill_betweenx([i - samp_bandwidth_height / 2, i + samp_bandwidth_height / 2], rf_band_center_MHz - samp_bandwidth_MHz / 2, rf_band_center_MHz + samp_bandwidth_MHz / 2, color="r", alpha=0.3)
        

    ax1.set_yticks(range(len(metadata.channel_ids)))
    ax1.set_yticklabels(metadata.channel_ids)
    for ax in [ax1, ax2]:
        ax.grid()
    ax1.set_xlabel("Baseband Frequency [MHz]")
    ax2.set_xlabel("RF Frequency [MHz]")