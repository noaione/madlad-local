use std::io::Write;

use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use clap::builder::styling::{AnsiColor, Effects};
use clap::builder::Styles;
use clap::Parser;
use languages::LanguageCode;

mod languages;
mod models;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None, styles = cli_styles())]
struct ArgsCli {
    /// The text input
    input: String,
    /// The target language
    #[arg(short, long)]
    target: LanguageCode,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The device to use, default to GPU if available
    #[arg(long, default_value = "gpu")]
    device: models::DeviceUse,
}

fn main() -> anyhow::Result<()> {
    let args = ArgsCli::parse();

    println!("✨✨✨✨ MADLAD-400-3B-MT ✨✨✨✨\n");
    let (builder, mut tokenizer) = models::MADLAD400Model::load(args.device)?;
    let device = builder.device();
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;

    let encoded_string = format!("<2{}>{}", args.target.to_token(), &args.input);

    println!("[?] Tokenizing input...");
    let tokens = tokenizer
        .encode(encoded_string, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    println!("[?] Creating input tensor...");
    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    println!("[?] Building model...");
    let mut model = builder.build_model()?;
    println!("[?] Creating output tensor...");
    let mut output_token_ids = [builder
        .config()
        .decoder_start_token_id
        .unwrap_or(builder.config().pad_token_id) as u32]
    .to_vec();

    let temperature = if args.temperature <= 0. {
        None
    } else {
        Some(args.temperature)
    };

    println!("[?] Creating logits processor...");
    let mut logits_processor = LogitsProcessor::new(299792458, temperature, args.top_p);
    println!("[?] Encoding input tokens...");
    let encoder_output = model.encode(&input_token_ids)?;
    let start = std::time::Instant::now();

    println!("\n[?] Inferencing...");
    println!("[!] Input: {}", &args.input);
    print!("[!] Output:");
    for index in 0.. {
        if output_token_ids.len() > 512 {
            break;
        }
        let decoder_token_ids = if index == 0 || !builder.config().use_cache {
            Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };
        let logits = model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &output_token_ids[start_at..],
            )?
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == builder.config().eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }

    let dt = start.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        output_token_ids.len(),
        output_token_ids.len() as f64 / dt.as_secs_f64(),
    );

    Ok(())
}

fn cli_styles() -> Styles {
    Styles::styled()
        .header(AnsiColor::Green.on_default() | Effects::BOLD)
        .usage(AnsiColor::Magenta.on_default() | Effects::BOLD | Effects::UNDERLINE)
        .literal(AnsiColor::Blue.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::BrightCyan.on_default())
}
