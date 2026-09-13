import click
import yaml
import os
import sys
import importlib

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.utils.prepare_data import RealDataLoader
from generators.models.multivariate import MultivariateDataGenerator
from generators.utils.resource import ResourceProfiler


generator_classes = {
    'multivariate': ('models.multivariate', 'MultivariateDataGenerator'),
    'cvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    'dpcvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    'ctgan': ('models.sdv_ctgan', 'CTGANDataGenerationPipeline'),
    'dpctgan': ('models.dpctgan', 'DPCTGANDataGenerationPipeline'),
    'sc_dist': ('models.sc_dist', 'ScDistributionDataGenerator'),
    'cvae_gmm': ('models.cvae_gmm', 'CVAEGMMDataGenerator'),
    'wgan_gp': ('models.wgan_gp', 'WGANGPDataGenerator'),
    # Bulk DP-NMF generator (adapted from the single-cell nmf_sampler concept)
    'nmf_sampler': ('models.nmf_sampler', 'NMFSamplerDataGenerator'),
}


## dynamic import to avoid package versioning errors
def get_generator_class(generator_name):
    if generator_name in generator_classes:
        module_name, class_name = generator_classes[generator_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Unknown generator name: {generator_name}")


@click.group()
def cli():
    pass


## a stratified 5 fold CV split will be created under
## data_splits/split_indices/{dataset_name}_splits.yaml
## update random_seed in dataset_config to generate an original split
@click.command()
def generate_split_indices():
    config = yaml.safe_load(open("config.yaml"))
    rdataloader = RealDataLoader(config)
    rdataloader.save_split_indices()


## the real data will be split into 5 train/test pairs
## based on the above generated {dataset_name}_splits.yaml
## the data will be saved under data_splits/{dataset_name}/real/
@click.command()
def generate_data_splits():
    config = yaml.safe_load(open("config.yaml"))
    rdataloader = RealDataLoader(config)
    rdataloader.save_split_data()


## your synthetic data will be saved accordingly to config.yaml
## e.g. data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}
## resource usage (train/generation time, peak memory, params) is logged to
## resource_logs/{experiment_name}_split{split_no}.json
@click.command()
@click.argument('split_no', type=int)
@click.option('--experiment_name', type=str, default="")
def run_generator(split_no: int, experiment_name: str = None):
    config = yaml.safe_load(open("config.yaml"))
    generator_name = config.get("generator_name")

    prof = ResourceProfiler()

    GeneratorClass = get_generator_class(generator_name)
    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    metrics = {
        "model": generator_name,
        "split_no": split_no,
        "experiment_name": experiment_name,
        **prof.get_gpu_info(),
        **prof.get_cpu_info(),
        "train_time_sec": None,
        "generation_time_sec": None,
        "n_generated_samples": None,
        "n_parameters_M": None,
        "train_peak_memory_gb": None,
        "gen_peak_memory_gb": None,
    }

    # Multivariate has no training phase: profile generation only.
    if isinstance(generator, MultivariateDataGenerator):
        if config.get("generate", True):
            prof.start()
            syn_data, syn_lbl = generator.generate()
            gen_stats = prof.stop()
            metrics["generation_time_sec"] = gen_stats["elapsed_sec"]
            metrics["gen_peak_memory_gb"] = gen_stats["peak_memory_gb"]
            metrics["n_generated_samples"] = len(syn_data)
            generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)
        prof.save_metrics(metrics, split_no, experiment_name)
        return

    # Trainable generators (incl. nmf_sampler): profile train and generation separately.
    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            prof.start()
            generator.train()
            train_stats = prof.stop()
            metrics["train_time_sec"] = train_stats["elapsed_sec"]
            metrics["train_peak_memory_gb"] = train_stats["peak_memory_gb"]
    else:
        generator.load_from_checkpoint()

    if hasattr(generator, "model"):
        metrics["n_parameters_M"] = prof.count_parameters(generator.model)

    if config.get("generate", True):
        prof.start()
        syn_data, syn_lbl = generator.generate()
        gen_stats = prof.stop()
        metrics["generation_time_sec"] = gen_stats["elapsed_sec"]
        metrics["gen_peak_memory_gb"] = gen_stats["peak_memory_gb"]
        metrics["n_generated_samples"] = len(syn_data)
        generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)

    prof.save_metrics(metrics, split_no, experiment_name)


cli.add_command(generate_split_indices)
cli.add_command(generate_data_splits)
cli.add_command(run_generator)

if __name__ == '__main__':
    cli()
