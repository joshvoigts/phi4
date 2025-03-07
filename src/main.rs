use anyhow::Result;

mod phi4;

const MODELS_PATH: &str =
  concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const TOKENIZER_PATH: &str =
  concat!(env!("CARGO_MANIFEST_DIR"), "/models/tokenizer.json");

fn main() -> Result<()> {
  tracing_subscriber::fmt::init();

  // Create the ONNX Runtime environment
  let environment = ort::init().commit()?;

  // Initialize the Phi4MMProcessor
  let processor = phi4::Phi4MMProcessor::new(
    MODELS_PATH,
    TOKENIZER_PATH,
    environment,
  )?;

  // Example text to process
  let result = processor.process("Hello, world!", None, None)?;
  println!("Model output: {}", result);

  Ok(())
}
