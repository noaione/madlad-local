use std::path::PathBuf;

use candle_core::Device;
use candle_transformers::models::quantized_t5 as t5;
use clap::ValueEnum;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::Repo;
use tokenizers::Tokenizer;

#[cfg(feature = "model-10b")]
pub(crate) const MODEL_ID: &str = "jbochi/madlad400-10b-mt";
#[cfg(feature = "model-7b-bt")]
pub(crate) const MODEL_ID: &str = "jbochi/madlad400-7b-mt-bt";
#[cfg(feature = "model-7b")]
pub(crate) const MODEL_ID: &str = "jbochi/madlad400-7b-mt";
// Everything elses
#[cfg(not(any(feature = "model-10b", feature = "model-7b-bt", feature = "model-7b")))]
pub(crate) const MODEL_ID: &str = "jbochi/madlad400-3b-mt";

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub(crate) enum DeviceUse {
    /// Use CPU mode
    CPU,
    /// Use GPU/CUDA mode if available, fallback to CPU mode otherwise
    #[default]
    GPU,
}
pub struct MADLAD400Model {
    device: Device,
    config: t5::Config,
    weights: PathBuf,
}

impl MADLAD400Model {
    pub fn load(input_device: DeviceUse) -> anyhow::Result<(Self, Tokenizer)> {
        let device = match input_device {
            DeviceUse::GPU => Device::cuda_if_available(0)?,
            DeviceUse::CPU => Device::Cpu,
        };

        let repo = Repo::with_revision(
            MODEL_ID.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        );
        let api = Api::new()?;
        // Download the model and everything to {CWD}/models
        let api = api.repo(repo);
        let config_filename = Self::get_local_or_remote_file("config.json", &api)?;
        println!("[?] Loading config from {}", config_filename.display());
        let tokenizer_filename = Self::get_local_or_remote_file("tokenizer.json", &api)?;
        println!(
            "[?] Loading tokenizer from {}",
            tokenizer_filename.display()
        );
        let weights_filename = Self::get_local_or_remote_file("model-q4k.gguf", &api)?;
        println!("[?] Loading weights from {}", weights_filename.display());

        let configs = std::fs::read_to_string(&config_filename)?;
        let config: t5::Config = serde_json::from_str(&configs)?;
        println!("[?] Loaded config: {}", config_filename.display());
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;
        println!("[?] Loaded tokenizer: {}", tokenizer_filename.display());

        Ok((
            Self {
                device,
                config,
                weights: weights_filename,
            },
            tokenizer,
        ))
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> anyhow::Result<PathBuf> {
        let cwd = std::env::current_dir()?;
        let local_filename = cwd.join("models").join(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &t5::Config {
        &self.config
    }

    pub fn build_model(&self) -> anyhow::Result<t5::T5ForConditionalGeneration> {
        let vb = t5::VarBuilder::from_gguf(&self.weights, &self.device)?;
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }
}
